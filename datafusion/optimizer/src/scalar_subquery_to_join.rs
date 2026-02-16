// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`ScalarSubqueryToJoin`] rewriting scalar subquery filters to `JOIN`s

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::decorrelate::{PullUpCorrelatedExpr, UN_MATCHED_ROW_INDICATOR};
use crate::optimizer::ApplyOrder;
use crate::utils::{evaluates_to_null, replace_qualified_name};
use crate::{OptimizerConfig, OptimizerRule};

use crate::analyzer::type_coercion::TypeCoercionRewriter;
use datafusion_common::alias::AliasGenerator;
use datafusion_common::tree_node::{
    Transformed, TransformedResult, TreeNode, TreeNodeRecursion, TreeNodeRewriter,
};
use datafusion_common::{
    assert_or_internal_err, plan_err, Column, DataFusionError, Result, ScalarValue,
};
use datafusion_expr::expr_rewriter::create_col_from_scalar_expr;
use datafusion_expr::logical_plan::{JoinType, Subquery};
use datafusion_expr::utils::{conjunction, split_conjunction};
use datafusion_expr::{
    binary_expr, expr, EmptyRelation, Expr, LogicalPlan, LogicalPlanBuilder, Operator,
};

/// Optimizer rule for rewriting subquery filters to joins
/// and places additional projection on top of the filter, to preserve
/// original schema.
#[derive(Default, Debug)]
pub struct ScalarSubqueryToJoin {}

impl ScalarSubqueryToJoin {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Finds expressions that have a scalar subquery in them (and recurses when found)
    ///
    /// # Arguments
    /// * `predicate` - A conjunction to split and search
    ///
    /// Returns a tuple (subqueries, alias)
    fn extract_subquery_exprs(
        &self,
        predicate: &Expr,
        alias_gen: &Arc<AliasGenerator>,
    ) -> Result<(Vec<(Subquery, String)>, Expr)> {
        let mut extract = ExtractScalarSubQuery {
            sub_query_info: vec![],
            alias_gen,
        };
        predicate
            .clone()
            .rewrite(&mut extract)
            .data()
            .map(|new_expr| (extract.sub_query_info, new_expr))
    }
}

impl OptimizerRule for ScalarSubqueryToJoin {
    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        match plan {
            LogicalPlan::Filter(filter) => {
                // Optimization: skip the rest of the rule and its copies if
                // there are no scalar subqueries
                if !contains_scalar_subquery(&filter.predicate) {
                    return Ok(Transformed::no(LogicalPlan::Filter(filter)));
                }

                let (subqueries, mut rewrite_expr) = self.extract_subquery_exprs(
                    &filter.predicate,
                    config.alias_generator(),
                )?;

                assert_or_internal_err!(
                    !subqueries.is_empty(),
                    "Expected subqueries not found in filter"
                );

                // iterate through all subqueries in predicate, turning each into a left join
                let mut cur_input = filter.input.as_ref().clone();
                for (subquery, alias) in subqueries {
                    if let Some((optimized_subquery, expr_check_map)) =
                        build_join(&subquery, &cur_input, &alias)?
                    {
                        if !expr_check_map.is_empty() {
                            rewrite_expr = rewrite_expr
                                .transform_up(|expr| {
                                    // replace column references with entry in map, if it exists
                                    if let Some(map_expr) = expr
                                        .try_as_col()
                                        .and_then(|col| expr_check_map.get(&col.name))
                                    {
                                        Ok(Transformed::yes(map_expr.clone()))
                                    } else {
                                        Ok(Transformed::no(expr))
                                    }
                                })
                                .data()?;
                        }
                        cur_input = optimized_subquery;
                    } else {
                        // if we can't handle all of the subqueries then bail for now
                        return Ok(Transformed::no(LogicalPlan::Filter(filter)));
                    }
                }

                // Preserve original schema as new Join might have more fields than what Filter & parents expect.
                let projection =
                    filter.input.schema().columns().into_iter().map(Expr::from);
                let new_plan = LogicalPlanBuilder::from(cur_input)
                    .filter(rewrite_expr)?
                    .project(projection)?
                    .build()?;
                Ok(Transformed::yes(new_plan))
            }
            LogicalPlan::Projection(projection) => {
                // Optimization: skip the rest of the rule and its copies if
                // there are no scalar subqueries
                if !projection.expr.iter().any(contains_scalar_subquery) {
                    return Ok(Transformed::no(LogicalPlan::Projection(projection)));
                }

                let mut all_subqueries = vec![];
                let mut expr_to_rewrite_expr_map = HashMap::new();
                let mut subquery_to_expr_map = HashMap::new();
                for expr in projection.expr.iter() {
                    let (subqueries, rewrite_exprs) =
                        self.extract_subquery_exprs(expr, config.alias_generator())?;
                    for (subquery, _) in &subqueries {
                        subquery_to_expr_map.insert(subquery.clone(), expr.clone());
                    }
                    all_subqueries.extend(subqueries);
                    expr_to_rewrite_expr_map.insert(expr, rewrite_exprs);
                }
                assert_or_internal_err!(
                    !all_subqueries.is_empty(),
                    "Expected subqueries not found in projection"
                );
                // iterate through all subqueries in predicate, turning each into a left join
                let mut cur_input = projection.input.as_ref().clone();
                for (subquery, alias) in all_subqueries {
                    if let Some((optimized_subquery, expr_check_map)) =
                        build_join(&subquery, &cur_input, &alias)?
                    {
                        cur_input = optimized_subquery;
                        if !expr_check_map.is_empty() {
                            if let Some(expr) = subquery_to_expr_map.get(&subquery) {
                                if let Some(rewrite_expr) =
                                    expr_to_rewrite_expr_map.get(expr)
                                {
                                    let new_expr = rewrite_expr
                                        .clone()
                                        .transform_up(|expr| {
                                            // replace column references with entry in map, if it exists
                                            if let Some(map_expr) =
                                                expr.try_as_col().and_then(|col| {
                                                    expr_check_map.get(&col.name)
                                                })
                                            {
                                                Ok(Transformed::yes(map_expr.clone()))
                                            } else {
                                                Ok(Transformed::no(expr))
                                            }
                                        })
                                        .data()?;
                                    expr_to_rewrite_expr_map.insert(expr, new_expr);
                                }
                            }
                        }
                    } else {
                        // if we can't handle all of the subqueries then bail for now
                        return Ok(Transformed::no(LogicalPlan::Projection(projection)));
                    }
                }

                let mut proj_exprs = vec![];
                for expr in projection.expr.iter() {
                    let old_expr_name = expr.schema_name().to_string();
                    let new_expr = expr_to_rewrite_expr_map.get(expr).unwrap();
                    let new_expr_name = new_expr.schema_name().to_string();
                    if new_expr_name != old_expr_name {
                        proj_exprs.push(new_expr.clone().alias(old_expr_name))
                    } else {
                        proj_exprs.push(new_expr.clone());
                    }
                }
                let new_plan = LogicalPlanBuilder::from(cur_input)
                    .project(proj_exprs)?
                    .build()?;
                Ok(Transformed::yes(new_plan))
            }

            plan => Ok(Transformed::no(plan)),
        }
    }

    fn name(&self) -> &str {
        "scalar_subquery_to_join"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }
}

/// Returns true if the expression has a scalar subquery somewhere in it
/// false otherwise
fn contains_scalar_subquery(expr: &Expr) -> bool {
    expr.exists(|expr| Ok(matches!(expr, Expr::ScalarSubquery(_))))
        .expect("Inner is always Ok")
}

struct ExtractScalarSubQuery<'a> {
    sub_query_info: Vec<(Subquery, String)>,
    alias_gen: &'a Arc<AliasGenerator>,
}

impl TreeNodeRewriter for ExtractScalarSubQuery<'_> {
    type Node = Expr;

    fn f_down(&mut self, expr: Expr) -> Result<Transformed<Expr>> {
        match expr {
            Expr::ScalarSubquery(subquery) => {
                let subqry_alias = self.alias_gen.next("__scalar_sq");
                self.sub_query_info
                    .push((subquery.clone(), subqry_alias.clone()));
                let scalar_expr = subquery
                    .subquery
                    .head_output_expr()?
                    .map_or(plan_err!("single expression required."), Ok)?;
                Ok(Transformed::new(
                    Expr::Column(create_col_from_scalar_expr(
                        &scalar_expr,
                        subqry_alias,
                    )?),
                    true,
                    TreeNodeRecursion::Jump,
                ))
            }
            _ => Ok(Transformed::no(expr)),
        }
    }
}

/// Takes a query like:
///
/// ```text
/// select id from customers where balance >
///     (select avg(total) from orders where orders.c_id = customers.id)
/// ```
///
/// and optimizes it into:
///
/// ```text
/// select c.id from customers c
/// left join (select c_id, avg(total) as val from orders group by c_id) o on o.c_id = c.c_id
/// where c.balance > o.val
/// ```
///
/// Or a query like:
///
/// ```text
/// select id from customers where balance >
///     (select avg(total) from orders)
/// ```
///
/// and optimizes it into:
///
/// ```text
/// select c.id from customers c
/// left join (select avg(total) as val from orders) a
/// where c.balance > a.val
/// ```
///
/// # Arguments
///
/// * `query_info` - The subquery portion of the `where` (select avg(total) from orders)
/// * `filter_input` - The non-subquery portion (from customers)
/// * `outer_others` - Any additional parts to the `where` expression (and c.x = y)
/// * `subquery_alias` - Subquery aliases
fn build_join(
    subquery: &Subquery,
    filter_input: &LogicalPlan,
    subquery_alias: &str,
) -> Result<Option<(LogicalPlan, HashMap<String, Expr>)>> {
    let subquery_plan = subquery.subquery.as_ref();
    let mut pull_up = PullUpCorrelatedExpr::new().with_need_handle_count_bug(true);
    let new_plan = subquery_plan.clone().rewrite(&mut pull_up).data()?;
    if !pull_up.can_pull_up {
        return build_join_with_domain_keys(subquery_plan, filter_input, subquery_alias);
    }

    let collected_count_expr_map =
        pull_up.collected_count_expr_map.get(&new_plan).cloned();
    let sub_query_alias = LogicalPlanBuilder::from(new_plan)
        .alias(subquery_alias.to_string())?
        .build()?;

    let mut all_correlated_cols = BTreeSet::new();
    pull_up
        .correlated_subquery_cols_map
        .values()
        .for_each(|cols| all_correlated_cols.extend(cols.clone()));

    // alias the join filter
    let join_filter_opt =
        conjunction(pull_up.join_filters).map_or(Ok(None), |filter| {
            replace_qualified_name(filter, &all_correlated_cols, subquery_alias).map(Some)
        })?;

    // join our sub query into the main plan
    let new_plan = if join_filter_opt.is_none() {
        match filter_input {
            LogicalPlan::EmptyRelation(EmptyRelation {
                produce_one_row: true,
                schema: _,
            }) => sub_query_alias,
            _ => {
                // if not correlated, group down to 1 row and left join on that (preserving row count)
                LogicalPlanBuilder::from(filter_input.clone())
                    .join_on(
                        sub_query_alias,
                        JoinType::Left,
                        vec![Expr::Literal(ScalarValue::Boolean(Some(true)), None)],
                    )?
                    .build()?
            }
        }
    } else {
        // left join if correlated, grouping by the join keys so we don't change row count
        LogicalPlanBuilder::from(filter_input.clone())
            .join_on(sub_query_alias, JoinType::Left, join_filter_opt)?
            .build()?
    };
    let mut computation_project_expr = HashMap::new();
    if let Some(expr_map) = collected_count_expr_map {
        for (name, result) in expr_map {
            if evaluates_to_null(result.clone(), result.column_refs())? {
                // If expr always returns null when column is null, skip processing
                continue;
            }
            let computer_expr = if let Some(filter) = &pull_up.pull_up_having_expr {
                Expr::Case(expr::Case {
                    expr: None,
                    when_then_expr: vec![
                        (
                            Box::new(Expr::IsNull(Box::new(Expr::Column(
                                Column::new_unqualified(UN_MATCHED_ROW_INDICATOR),
                            )))),
                            Box::new(result),
                        ),
                        (
                            Box::new(Expr::Not(Box::new(filter.clone()))),
                            Box::new(Expr::Literal(ScalarValue::Null, None)),
                        ),
                    ],
                    else_expr: Some(Box::new(Expr::Column(Column::new_unqualified(
                        name.clone(),
                    )))),
                })
            } else {
                Expr::Case(expr::Case {
                    expr: None,
                    when_then_expr: vec![(
                        Box::new(Expr::IsNull(Box::new(Expr::Column(
                            Column::new_unqualified(UN_MATCHED_ROW_INDICATOR),
                        )))),
                        Box::new(result),
                    )],
                    else_expr: Some(Box::new(Expr::Column(Column::new_unqualified(
                        name.clone(),
                    )))),
                })
            };
            let mut expr_rewrite = TypeCoercionRewriter {
                schema: new_plan.schema(),
            };
            computation_project_expr
                .insert(name, computer_expr.rewrite(&mut expr_rewrite).data()?);
        }
    }

    Ok(Some((new_plan, computation_project_expr)))
}

/// Fallback rewrite for correlated scalar aggregates that `PullUpCorrelatedExpr` currently
/// cannot decorrelate (for example non-equality/disjunctive correlated predicates).
///
/// This implements a constrained domain-key rewrite:
/// - required root shape: Projection -> [local Filter]* -> Aggregate
/// - aggregate input supports local Projection/Filter layers above one correlated Filter
/// - the only outer references must be in that correlated Filter predicate
fn build_join_with_domain_keys(
    subquery_plan: &LogicalPlan,
    filter_input: &LogicalPlan,
    subquery_alias: &str,
) -> Result<Option<(LogicalPlan, HashMap<String, Expr>)>> {
    let LogicalPlan::Projection(top_projection) = subquery_plan else {
        return Ok(None);
    };
    if top_projection.expr.iter().any(Expr::contains_outer) {
        return Ok(None);
    }

    let mut post_aggregate_filters = vec![];
    let mut aggregate_plan = top_projection.input.as_ref();
    while let LogicalPlan::Filter(filter) = aggregate_plan {
        if filter.predicate.contains_outer() {
            return Ok(None);
        }
        post_aggregate_filters.push(filter.predicate.clone());
        aggregate_plan = filter.input.as_ref();
    }

    let LogicalPlan::Aggregate(aggregate) = aggregate_plan else {
        return Ok(None);
    };
    if aggregate.group_expr.iter().any(Expr::contains_outer)
        || aggregate.aggr_expr.iter().any(Expr::contains_outer)
    {
        return Ok(None);
    }

    let mut pre_aggregate_unary_nodes = vec![];
    let mut aggregate_input_plan = aggregate.input.as_ref();
    let correlated_filter = loop {
        match aggregate_input_plan {
            LogicalPlan::Filter(filter) => {
                if filter.predicate.contains_outer() {
                    break filter;
                }
                pre_aggregate_unary_nodes
                    .push(DomainRewriteUnaryNode::Filter(filter.predicate.clone()));
                aggregate_input_plan = filter.input.as_ref();
            }
            LogicalPlan::Projection(projection) => {
                if projection.expr.iter().any(Expr::contains_outer) {
                    return Ok(None);
                }
                pre_aggregate_unary_nodes
                    .push(DomainRewriteUnaryNode::Projection(projection.expr.clone()));
                aggregate_input_plan = projection.input.as_ref();
            }
            _ => return Ok(None),
        }
    };
    if !correlated_filter.predicate.contains_outer()
        || correlated_filter.input.contains_outer_reference()
    {
        return Ok(None);
    }
    let preserve_domain_relation = !pre_aggregate_unary_nodes
        .iter()
        .any(|node| matches!(node, DomainRewriteUnaryNode::Projection(_)));

    let mut empty_batch_expr_map = HashMap::new();
    for (name, expr) in
        projection_empty_batch_results(&aggregate.aggr_expr, &top_projection.expr)?
    {
        if !evaluates_to_null(expr.clone(), expr.column_refs())? {
            empty_batch_expr_map.insert(name, expr);
        }
    }
    // Post-aggregate filters (HAVING) can suppress the aggregate row, so nulls from
    // left join are ambiguous with empty-input cases. Keep this conservative until
    // count+having pull-up is implemented for the domain fallback.
    if !empty_batch_expr_map.is_empty() && !post_aggregate_filters.is_empty() {
        return Ok(None);
    }

    let outer_cols = collect_outer_reference_columns(subquery_plan);
    if outer_cols.is_empty() {
        return Ok(None);
    }

    let domain_alias = format!("{subquery_alias}_domain");
    let mut outer_col_to_domain_col = HashMap::new();
    let mut domain_project_exprs = Vec::with_capacity(outer_cols.len());
    let mut domain_col_names = Vec::with_capacity(outer_cols.len());
    for (idx, outer_col) in outer_cols.iter().enumerate() {
        let domain_col_name = format!("__correlated_sq_col_{idx}");
        outer_col_to_domain_col.insert(outer_col.clone(), domain_col_name.clone());
        domain_col_names.push(domain_col_name.clone());
        domain_project_exprs.push(Expr::Column(outer_col.clone()).alias(domain_col_name));
    }

    let domain_plan = LogicalPlanBuilder::from(filter_input.clone())
        .project(domain_project_exprs)?
        .distinct()?
        .alias(domain_alias.clone())?
        .build()?;

    let (correlated_filters, local_filters): (Vec<_>, Vec<_>) =
        split_conjunction(&correlated_filter.predicate)
            .into_iter()
            .partition(|expr| expr.contains_outer());
    if correlated_filters.is_empty() {
        return Ok(None);
    }

    let rewritten_correlated_filters = correlated_filters
        .into_iter()
        .map(|expr| {
            replace_outer_refs_with_domain_cols(
                expr.clone(),
                &outer_col_to_domain_col,
                &domain_alias,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    if rewritten_correlated_filters
        .iter()
        .any(Expr::contains_outer)
    {
        return Ok(None);
    }

    let mut rewritten_input = LogicalPlanBuilder::from(domain_plan)
        .join_on(
            correlated_filter.input.as_ref().clone(),
            JoinType::Inner,
            rewritten_correlated_filters,
        )?
        .build()?;

    if let Some(local_filter) =
        conjunction(local_filters.into_iter().cloned().collect::<Vec<_>>())
    {
        rewritten_input = LogicalPlanBuilder::from(rewritten_input)
            .filter(local_filter)?
            .build()?;
    }

    for unary_node in pre_aggregate_unary_nodes.into_iter().rev() {
        rewritten_input = match unary_node {
            DomainRewriteUnaryNode::Filter(predicate) => {
                LogicalPlanBuilder::from(rewritten_input)
                    .filter(predicate)?
                    .build()?
            }
            DomainRewriteUnaryNode::Projection(exprs) => {
                let mut projection_exprs = exprs;
                for domain_col_name in &domain_col_names {
                    if projection_exprs
                        .iter()
                        .any(|expr| expr_output_name(expr) == *domain_col_name)
                    {
                        continue;
                    }
                    projection_exprs.push(
                        Expr::Column(Column::new(
                            Some(domain_alias.clone()),
                            domain_col_name.clone(),
                        ))
                        .alias(domain_col_name.clone()),
                    );
                }
                LogicalPlanBuilder::from(rewritten_input)
                    .project(projection_exprs)?
                    .build()?
            }
        };
    }

    let mut group_expr = aggregate.group_expr.clone();
    for domain_col_name in &domain_col_names {
        let domain_group_col = if preserve_domain_relation {
            Expr::Column(Column::new(
                Some(domain_alias.clone()),
                domain_col_name.clone(),
            ))
        } else {
            Expr::Column(Column::new_unqualified(domain_col_name.clone()))
        };
        group_expr.push(domain_group_col.alias(domain_col_name.clone()));
    }

    let rewritten_aggregate = LogicalPlanBuilder::from(rewritten_input)
        .aggregate(group_expr, aggregate.aggr_expr.clone())?
        .build()?;

    let mut rewritten_subquery_input = rewritten_aggregate;
    for predicate in post_aggregate_filters.into_iter().rev() {
        rewritten_subquery_input = LogicalPlanBuilder::from(rewritten_subquery_input)
            .filter(predicate)?
            .build()?;
    }

    let mut rewritten_projection_exprs = top_projection.expr.clone();
    for domain_col_name in &domain_col_names {
        rewritten_projection_exprs.push(Expr::Column(Column::new_unqualified(
            domain_col_name.clone(),
        )));
    }
    if !empty_batch_expr_map.is_empty() {
        rewritten_projection_exprs.push(
            Expr::Literal(ScalarValue::Boolean(Some(true)), None)
                .alias(UN_MATCHED_ROW_INDICATOR),
        );
    }

    let rewritten_subquery = LogicalPlanBuilder::from(rewritten_subquery_input)
        .project(rewritten_projection_exprs)?
        .alias(subquery_alias.to_string())?
        .build()?;

    // Match outer rows to the corresponding domain-key aggregate row.
    let domain_join_filters = outer_cols
        .into_iter()
        .zip(domain_col_names)
        .map(|(outer_col, domain_col_name)| {
            binary_expr(
                Expr::Column(outer_col),
                Operator::IsNotDistinctFrom,
                Expr::Column(Column::new(
                    Some(subquery_alias.to_string()),
                    domain_col_name,
                )),
            )
        })
        .collect::<Vec<_>>();

    let rewritten_plan = LogicalPlanBuilder::from(filter_input.clone())
        .join_on(rewritten_subquery, JoinType::Left, domain_join_filters)?
        .build()?;

    let mut computation_project_expr = HashMap::new();
    for (name, result) in empty_batch_expr_map {
        let computer_expr = Expr::Case(expr::Case {
            expr: None,
            when_then_expr: vec![(
                Box::new(Expr::IsNull(Box::new(Expr::Column(
                    Column::new_unqualified(UN_MATCHED_ROW_INDICATOR),
                )))),
                Box::new(result),
            )],
            else_expr: Some(Box::new(Expr::Column(Column::new_unqualified(
                name.clone(),
            )))),
        });
        let mut expr_rewrite = TypeCoercionRewriter {
            schema: rewritten_plan.schema(),
        };
        computation_project_expr
            .insert(name, computer_expr.rewrite(&mut expr_rewrite).data()?);
    }

    Ok(Some((rewritten_plan, computation_project_expr)))
}

fn collect_outer_reference_columns(plan: &LogicalPlan) -> Vec<Column> {
    let mut outer_cols = vec![];
    for expr in plan.all_out_ref_exprs() {
        if let Expr::OuterReferenceColumn(_, col) = expr {
            if !outer_cols.contains(&col) {
                outer_cols.push(col);
            }
        }
    }
    outer_cols
}

fn replace_outer_refs_with_domain_cols(
    expr: Expr,
    outer_col_to_domain_col: &HashMap<Column, String>,
    domain_alias: &str,
) -> Result<Expr> {
    expr.transform_up(|expr| match expr {
        Expr::OuterReferenceColumn(_, col) => {
            if let Some(domain_col) = outer_col_to_domain_col.get(&col) {
                Ok(Transformed::yes(Expr::Column(Column::new(
                    Some(domain_alias.to_string()),
                    domain_col.clone(),
                ))))
            } else {
                plan_err!(
                    "Can not map correlated column {col} into generated scalar-subquery domain"
                )
            }
        }
        _ => Ok(Transformed::no(expr)),
    })
    .data()
}

#[derive(Debug)]
enum DomainRewriteUnaryNode {
    Filter(Expr),
    Projection(Vec<Expr>),
}

fn projection_empty_batch_results(
    aggregate_exprs: &[Expr],
    projection_exprs: &[Expr],
) -> Result<HashMap<String, Expr>> {
    let mut aggregate_empty_results = HashMap::new();
    for aggregate_expr in aggregate_exprs {
        let result_expr = aggregate_expr
            .clone()
            .transform_up(|expr| {
                let new_expr = match expr {
                    Expr::AggregateFunction(expr::AggregateFunction { func, .. }) => {
                        if func.name() == "count" {
                            Transformed::yes(Expr::Literal(
                                ScalarValue::Int64(Some(0)),
                                None,
                            ))
                        } else {
                            Transformed::yes(Expr::Literal(ScalarValue::Null, None))
                        }
                    }
                    _ => Transformed::no(expr),
                };
                Ok(new_expr)
            })
            .data()?;
        aggregate_empty_results.insert(expr_output_name(aggregate_expr), result_expr);
    }

    let mut projection_empty_results = HashMap::new();
    for projection_expr in projection_exprs {
        let result_expr = projection_expr
            .clone()
            .transform_up(|expr| {
                let new_expr = match expr {
                    Expr::AggregateFunction(expr::AggregateFunction { func, .. }) => {
                        if func.name() == "count" {
                            Transformed::yes(Expr::Literal(
                                ScalarValue::Int64(Some(0)),
                                None,
                            ))
                        } else {
                            Transformed::yes(Expr::Literal(ScalarValue::Null, None))
                        }
                    }
                    Expr::Column(col) => {
                        let name = col.name.clone();
                        if let Some(result_expr) = aggregate_empty_results.get(&name) {
                            Transformed::yes(result_expr.clone())
                        } else {
                            Transformed::no(Expr::Column(col))
                        }
                    }
                    _ => Transformed::no(expr),
                };
                Ok(new_expr)
            })
            .data()?;
        if result_expr.ne(projection_expr) {
            projection_empty_results
                .insert(expr_output_name(projection_expr), result_expr.unalias());
        }
    }

    Ok(projection_empty_results)
}

fn expr_output_name(expr: &Expr) -> String {
    match expr {
        Expr::Alias(expr::Alias { name, .. }) => name.to_string(),
        Expr::Column(Column { name, .. }) => name.to_string(),
        _ => expr.schema_name().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Add;

    use super::*;
    use crate::test::*;

    use arrow::datatypes::DataType;
    use datafusion_expr::test::function_stub::sum;

    use crate::assert_optimized_plan_eq_display_indent_snapshot;
    use datafusion_expr::{col, lit, out_ref_col, scalar_subquery, Between};
    use datafusion_functions_aggregate::expr_fn::count;
    use datafusion_functions_aggregate::min_max::{max, min};

    macro_rules! assert_optimized_plan_equal {
        (
            $plan:expr,
            @ $expected:literal $(,)?
        ) => {{
            let rule: Arc<dyn crate::OptimizerRule + Send + Sync> = Arc::new(ScalarSubqueryToJoin::new());
            assert_optimized_plan_eq_display_indent_snapshot!(
                rule,
                $plan,
                @ $expected,
            )
        }};
    }

    /// Test multiple correlated subqueries
    #[test]
    fn multiple_subqueries() -> Result<()> {
        let orders = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    col("orders.o_custkey")
                        .eq(out_ref_col(DataType::Int64, "customer.c_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(
                lit(1)
                    .lt(scalar_subquery(Arc::clone(&orders)))
                    .and(lit(1).lt(scalar_subquery(orders))),
            )?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: Int32(1) < __scalar_sq_1.max(orders.o_custkey) AND Int32(1) < __scalar_sq_2.max(orders.o_custkey) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
              Left Join:  Filter: __scalar_sq_2.o_custkey = customer.c_custkey [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
                Left Join:  Filter: __scalar_sq_1.o_custkey = customer.c_custkey [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
                  TableScan: customer [c_custkey:Int64, c_name:Utf8]
                  SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                    Projection: max(orders.o_custkey), orders.o_custkey, __always_true [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                      Aggregate: groupBy=[[orders.o_custkey, Boolean(true) AS __always_true]], aggr=[[max(orders.o_custkey)]] [o_custkey:Int64, __always_true:Boolean, max(orders.o_custkey):Int64;N]
                        TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                SubqueryAlias: __scalar_sq_2 [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                  Projection: max(orders.o_custkey), orders.o_custkey, __always_true [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                    Aggregate: groupBy=[[orders.o_custkey, Boolean(true) AS __always_true]], aggr=[[max(orders.o_custkey)]] [o_custkey:Int64, __always_true:Boolean, max(orders.o_custkey):Int64;N]
                      TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test recursive correlated subqueries
    #[test]
    fn recursive_subqueries() -> Result<()> {
        let lineitem = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("lineitem"))
                .filter(
                    col("lineitem.l_orderkey")
                        .eq(out_ref_col(DataType::Int64, "orders.o_orderkey")),
                )?
                .aggregate(
                    Vec::<Expr>::new(),
                    vec![sum(col("lineitem.l_extendedprice"))],
                )?
                .project(vec![sum(col("lineitem.l_extendedprice"))])?
                .build()?,
        );

        let orders = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    col("orders.o_custkey")
                        .eq(out_ref_col(DataType::Int64, "customer.c_custkey"))
                        .and(col("orders.o_totalprice").lt(scalar_subquery(lineitem))),
                )?
                .aggregate(Vec::<Expr>::new(), vec![sum(col("orders.o_totalprice"))])?
                .project(vec![sum(col("orders.o_totalprice"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_acctbal").lt(scalar_subquery(orders)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_acctbal < __scalar_sq_1.sum(orders.o_totalprice) [c_custkey:Int64, c_name:Utf8, sum(orders.o_totalprice):Float64;N, o_custkey:Int64;N, __always_true:Boolean;N]
              Left Join:  Filter: __scalar_sq_1.o_custkey = customer.c_custkey [c_custkey:Int64, c_name:Utf8, sum(orders.o_totalprice):Float64;N, o_custkey:Int64;N, __always_true:Boolean;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [sum(orders.o_totalprice):Float64;N, o_custkey:Int64, __always_true:Boolean]
                  Projection: sum(orders.o_totalprice), orders.o_custkey, __always_true [sum(orders.o_totalprice):Float64;N, o_custkey:Int64, __always_true:Boolean]
                    Aggregate: groupBy=[[orders.o_custkey, Boolean(true) AS __always_true]], aggr=[[sum(orders.o_totalprice)]] [o_custkey:Int64, __always_true:Boolean, sum(orders.o_totalprice):Float64;N]
                      Projection: orders.o_orderkey, orders.o_custkey, orders.o_orderstatus, orders.o_totalprice [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                        Filter: orders.o_totalprice < __scalar_sq_2.sum(lineitem.l_extendedprice) [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N, sum(lineitem.l_extendedprice):Float64;N, l_orderkey:Int64;N, __always_true:Boolean;N]
                          Left Join:  Filter: __scalar_sq_2.l_orderkey = orders.o_orderkey [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N, sum(lineitem.l_extendedprice):Float64;N, l_orderkey:Int64;N, __always_true:Boolean;N]
                            TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                            SubqueryAlias: __scalar_sq_2 [sum(lineitem.l_extendedprice):Float64;N, l_orderkey:Int64, __always_true:Boolean]
                              Projection: sum(lineitem.l_extendedprice), lineitem.l_orderkey, __always_true [sum(lineitem.l_extendedprice):Float64;N, l_orderkey:Int64, __always_true:Boolean]
                                Aggregate: groupBy=[[lineitem.l_orderkey, Boolean(true) AS __always_true]], aggr=[[sum(lineitem.l_extendedprice)]] [l_orderkey:Int64, __always_true:Boolean, sum(lineitem.l_extendedprice):Float64;N]
                                  TableScan: lineitem [l_orderkey:Int64, l_partkey:Int64, l_suppkey:Int64, l_linenumber:Int32, l_quantity:Float64, l_extendedprice:Float64]
        "
        )
    }

    /// Test for correlated scalar subquery filter with additional subquery filters
    #[test]
    fn scalar_subquery_with_subquery_filters() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .eq(col("orders.o_custkey"))
                        .and(col("o_orderkey").eq(lit(1))),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").eq(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey = __scalar_sq_1.max(orders.o_custkey) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
              Left Join:  Filter: customer.c_custkey = __scalar_sq_1.o_custkey [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                  Projection: max(orders.o_custkey), orders.o_custkey, __always_true [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                    Aggregate: groupBy=[[orders.o_custkey, Boolean(true) AS __always_true]], aggr=[[max(orders.o_custkey)]] [o_custkey:Int64, __always_true:Boolean, max(orders.o_custkey):Int64;N]
                      Filter: orders.o_orderkey = Int32(1) [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                        TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test for correlated scalar subquery with no columns in schema
    #[test]
    fn scalar_subquery_no_cols() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .eq(out_ref_col(DataType::Int64, "customer.c_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").eq(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        // it will optimize, but fail for the same reason the unoptimized query would
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey = __scalar_sq_1.max(orders.o_custkey) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N]
              Left Join:  Filter: Boolean(true) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N]
                  Projection: max(orders.o_custkey) [max(orders.o_custkey):Int64;N]
                    Aggregate: groupBy=[[]], aggr=[[max(orders.o_custkey)]] [max(orders.o_custkey):Int64;N]
                      TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test for scalar subquery with both columns in schema
    #[test]
    fn scalar_subquery_with_no_correlated_cols() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(col("orders.o_custkey").eq(col("orders.o_custkey")))?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").eq(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey = __scalar_sq_1.max(orders.o_custkey) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N]
              Left Join:  Filter: Boolean(true) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N]
                  Projection: max(orders.o_custkey) [max(orders.o_custkey):Int64;N]
                    Aggregate: groupBy=[[]], aggr=[[max(orders.o_custkey)]] [max(orders.o_custkey):Int64;N]
                      Filter: orders.o_custkey = orders.o_custkey [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                        TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test for correlated scalar subquery not equal
    #[test]
    fn scalar_subquery_where_not_eq() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .not_eq(col("orders.o_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").eq(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        // Correlated non-equality scalar predicate can be decorrelated through domain keys.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey = __scalar_sq_1.max(orders.o_custkey) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64;N]
              Left Join:  Filter: customer.c_custkey IS NOT DISTINCT FROM __scalar_sq_1.__correlated_sq_col_0 [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64]
                  Projection: max(orders.o_custkey), __correlated_sq_col_0 [max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64]
                    Aggregate: groupBy=[[__scalar_sq_1_domain.__correlated_sq_col_0 AS __correlated_sq_col_0]], aggr=[[max(orders.o_custkey)]] [__correlated_sq_col_0:Int64, max(orders.o_custkey):Int64;N]
                      Inner Join:  Filter: __scalar_sq_1_domain.__correlated_sq_col_0 != orders.o_custkey [__correlated_sq_col_0:Int64, o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                        SubqueryAlias: __scalar_sq_1_domain [__correlated_sq_col_0:Int64]
                          Distinct: [__correlated_sq_col_0:Int64]
                            Projection: customer.c_custkey AS __correlated_sq_col_0 [__correlated_sq_col_0:Int64]
                              TableScan: customer [c_custkey:Int64, c_name:Utf8]
                        TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test for correlated scalar subquery less than
    #[test]
    fn scalar_subquery_where_less_than() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .lt(col("orders.o_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").eq(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        // Correlated non-equality scalar predicate can be decorrelated through domain keys.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey = __scalar_sq_1.max(orders.o_custkey) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64;N]
              Left Join:  Filter: customer.c_custkey IS NOT DISTINCT FROM __scalar_sq_1.__correlated_sq_col_0 [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64]
                  Projection: max(orders.o_custkey), __correlated_sq_col_0 [max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64]
                    Aggregate: groupBy=[[__scalar_sq_1_domain.__correlated_sq_col_0 AS __correlated_sq_col_0]], aggr=[[max(orders.o_custkey)]] [__correlated_sq_col_0:Int64, max(orders.o_custkey):Int64;N]
                      Inner Join:  Filter: __scalar_sq_1_domain.__correlated_sq_col_0 < orders.o_custkey [__correlated_sq_col_0:Int64, o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                        SubqueryAlias: __scalar_sq_1_domain [__correlated_sq_col_0:Int64]
                          Distinct: [__correlated_sq_col_0:Int64]
                            Projection: customer.c_custkey AS __correlated_sq_col_0 [__correlated_sq_col_0:Int64]
                              TableScan: customer [c_custkey:Int64, c_name:Utf8]
                        TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test for correlated scalar subquery not equal with count(*)
    #[test]
    fn scalar_subquery_where_not_eq_count() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .not_eq(col("orders.o_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![count(lit(1))])?
                .project(vec![count(lit(1))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").lt(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey < CASE WHEN __scalar_sq_1.__always_true IS NULL THEN Int64(0) ELSE __scalar_sq_1.count(Int32(1)) END [c_custkey:Int64, c_name:Utf8, count(Int32(1)):Int64;N, __correlated_sq_col_0:Int64;N, __always_true:Boolean;N]
              Left Join:  Filter: customer.c_custkey IS NOT DISTINCT FROM __scalar_sq_1.__correlated_sq_col_0 [c_custkey:Int64, c_name:Utf8, count(Int32(1)):Int64;N, __correlated_sq_col_0:Int64;N, __always_true:Boolean;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [count(Int32(1)):Int64, __correlated_sq_col_0:Int64, __always_true:Boolean]
                  Projection: count(Int32(1)), __correlated_sq_col_0, Boolean(true) AS __always_true [count(Int32(1)):Int64, __correlated_sq_col_0:Int64, __always_true:Boolean]
                    Aggregate: groupBy=[[__scalar_sq_1_domain.__correlated_sq_col_0 AS __correlated_sq_col_0]], aggr=[[count(Int32(1))]] [__correlated_sq_col_0:Int64, count(Int32(1)):Int64]
                      Inner Join:  Filter: __scalar_sq_1_domain.__correlated_sq_col_0 != orders.o_custkey [__correlated_sq_col_0:Int64, o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                        SubqueryAlias: __scalar_sq_1_domain [__correlated_sq_col_0:Int64]
                          Distinct: [__correlated_sq_col_0:Int64]
                            Projection: customer.c_custkey AS __correlated_sq_col_0 [__correlated_sq_col_0:Int64]
                              TableScan: customer [c_custkey:Int64, c_name:Utf8]
                        TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test for correlated scalar subquery filter with subquery disjunction
    #[test]
    fn scalar_subquery_with_subquery_disjunction() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .eq(col("orders.o_custkey"))
                        .or(col("o_orderkey").eq(lit(1))),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").eq(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        // Correlated disjunction scalar predicate can be decorrelated through domain keys.
        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey = __scalar_sq_1.max(orders.o_custkey) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64;N]
              Left Join:  Filter: customer.c_custkey IS NOT DISTINCT FROM __scalar_sq_1.__correlated_sq_col_0 [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64]
                  Projection: max(orders.o_custkey), __correlated_sq_col_0 [max(orders.o_custkey):Int64;N, __correlated_sq_col_0:Int64]
                    Aggregate: groupBy=[[__scalar_sq_1_domain.__correlated_sq_col_0 AS __correlated_sq_col_0]], aggr=[[max(orders.o_custkey)]] [__correlated_sq_col_0:Int64, max(orders.o_custkey):Int64;N]
                      Inner Join:  Filter: __scalar_sq_1_domain.__correlated_sq_col_0 = orders.o_custkey OR orders.o_orderkey = Int32(1) [__correlated_sq_col_0:Int64, o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                        SubqueryAlias: __scalar_sq_1_domain [__correlated_sq_col_0:Int64]
                          Distinct: [__correlated_sq_col_0:Int64]
                            Projection: customer.c_custkey AS __correlated_sq_col_0 [__correlated_sq_col_0:Int64]
                              TableScan: customer [c_custkey:Int64, c_name:Utf8]
                        TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test for correlated scalar subquery not equal with local unary layers
    #[test]
    fn scalar_subquery_where_not_eq_with_projection_and_having() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .not_eq(col("orders.o_custkey")),
                )?
                .project(vec![col("orders.o_custkey").alias("k")])?
                .aggregate(Vec::<Expr>::new(), vec![max(col("k"))])?
                .filter(col("max(k)").gt(lit(1)))?
                .project(vec![col("max(k)")])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").eq(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey = __scalar_sq_1.max(k) [c_custkey:Int64, c_name:Utf8, max(k):Int64;N, __correlated_sq_col_0:Int64;N]
              Left Join:  Filter: customer.c_custkey IS NOT DISTINCT FROM __scalar_sq_1.__correlated_sq_col_0 [c_custkey:Int64, c_name:Utf8, max(k):Int64;N, __correlated_sq_col_0:Int64;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(k):Int64;N, __correlated_sq_col_0:Int64]
                  Projection: max(k), __correlated_sq_col_0 [max(k):Int64;N, __correlated_sq_col_0:Int64]
                    Filter: max(k) > Int32(1) [__correlated_sq_col_0:Int64, max(k):Int64;N]
                      Aggregate: groupBy=[[__correlated_sq_col_0 AS __correlated_sq_col_0]], aggr=[[max(k)]] [__correlated_sq_col_0:Int64, max(k):Int64;N]
                        Projection: orders.o_custkey AS k, __scalar_sq_1_domain.__correlated_sq_col_0 AS __correlated_sq_col_0 [k:Int64, __correlated_sq_col_0:Int64]
                          Inner Join:  Filter: __scalar_sq_1_domain.__correlated_sq_col_0 != orders.o_custkey [__correlated_sq_col_0:Int64, o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                            SubqueryAlias: __scalar_sq_1_domain [__correlated_sq_col_0:Int64]
                              Distinct: [__correlated_sq_col_0:Int64]
                                Projection: customer.c_custkey AS __correlated_sq_col_0 [__correlated_sq_col_0:Int64]
                                  TableScan: customer [c_custkey:Int64, c_name:Utf8]
                            TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test for correlated scalar without projection
    #[test]
    fn scalar_subquery_no_projection() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(col("customer.c_custkey").eq(col("orders.o_custkey")))?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").eq(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        let expected = "Error during planning: Scalar subquery should only return one column, but found 4: orders.o_orderkey, orders.o_custkey, orders.o_orderstatus, orders.o_totalprice";
        assert_analyzer_check_err(vec![], plan, expected);
        Ok(())
    }

    /// Test for correlated scalar expressions
    #[test]
    fn scalar_subquery_project_expr() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .eq(col("orders.o_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![col("max(orders.o_custkey)").add(lit(1))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").eq(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey = __scalar_sq_1.max(orders.o_custkey) + Int32(1) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey) + Int32(1):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
              Left Join:  Filter: customer.c_custkey = __scalar_sq_1.o_custkey [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey) + Int32(1):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey) + Int32(1):Int64;N, o_custkey:Int64, __always_true:Boolean]
                  Projection: max(orders.o_custkey) + Int32(1), orders.o_custkey, __always_true [max(orders.o_custkey) + Int32(1):Int64;N, o_custkey:Int64, __always_true:Boolean]
                    Aggregate: groupBy=[[orders.o_custkey, Boolean(true) AS __always_true]], aggr=[[max(orders.o_custkey)]] [o_custkey:Int64, __always_true:Boolean, max(orders.o_custkey):Int64;N]
                      TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test for correlated scalar subquery with non-strong project
    #[test]
    fn scalar_subquery_with_non_strong_project() -> Result<()> {
        let case = Expr::Case(expr::Case {
            expr: None,
            when_then_expr: vec![(
                Box::new(col("max(orders.o_totalprice)")),
                Box::new(lit("a")),
            )],
            else_expr: Some(Box::new(lit("b"))),
        });

        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .eq(col("orders.o_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_totalprice"))])?
                .project(vec![case])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .project(vec![col("customer.c_custkey"), scalar_subquery(sq)])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r#"
        Projection: customer.c_custkey, CASE WHEN __scalar_sq_1.__always_true IS NULL THEN CASE WHEN CAST(NULL AS Boolean) THEN Utf8("a") ELSE Utf8("b") END ELSE __scalar_sq_1.CASE WHEN max(orders.o_totalprice) THEN Utf8("a") ELSE Utf8("b") END END AS CASE WHEN max(orders.o_totalprice) THEN Utf8("a") ELSE Utf8("b") END [c_custkey:Int64, CASE WHEN max(orders.o_totalprice) THEN Utf8("a") ELSE Utf8("b") END:Utf8;N]
          Left Join:  Filter: customer.c_custkey = __scalar_sq_1.o_custkey [c_custkey:Int64, c_name:Utf8, CASE WHEN max(orders.o_totalprice) THEN Utf8("a") ELSE Utf8("b") END:Utf8;N, o_custkey:Int64;N, __always_true:Boolean;N]
            TableScan: customer [c_custkey:Int64, c_name:Utf8]
            SubqueryAlias: __scalar_sq_1 [CASE WHEN max(orders.o_totalprice) THEN Utf8("a") ELSE Utf8("b") END:Utf8, o_custkey:Int64, __always_true:Boolean]
              Projection: CASE WHEN max(orders.o_totalprice) THEN Utf8("a") ELSE Utf8("b") END, orders.o_custkey, __always_true [CASE WHEN max(orders.o_totalprice) THEN Utf8("a") ELSE Utf8("b") END:Utf8, o_custkey:Int64, __always_true:Boolean]
                Aggregate: groupBy=[[orders.o_custkey, Boolean(true) AS __always_true]], aggr=[[max(orders.o_totalprice)]] [o_custkey:Int64, __always_true:Boolean, max(orders.o_totalprice):Float64;N]
                  TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "#
        )
    }

    /// Test for correlated scalar subquery multiple projected columns
    #[test]
    fn scalar_subquery_multi_col() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(col("customer.c_custkey").eq(col("orders.o_custkey")))?
                .project(vec![col("orders.o_custkey"), col("orders.o_orderkey")])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(
                col("customer.c_custkey")
                    .eq(scalar_subquery(sq))
                    .and(col("c_custkey").eq(lit(1))),
            )?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        let expected = "Error during planning: Scalar subquery should only return one column, but found 2: orders.o_custkey, orders.o_orderkey";
        assert_analyzer_check_err(vec![], plan, expected);
        Ok(())
    }

    /// Test for correlated scalar subquery filter with additional filters
    #[test]
    fn scalar_subquery_additional_filters_with_non_equal_clause() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .eq(col("orders.o_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(
                col("customer.c_custkey")
                    .gt_eq(scalar_subquery(sq))
                    .and(col("c_custkey").eq(lit(1))),
            )?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey >= __scalar_sq_1.max(orders.o_custkey) AND customer.c_custkey = Int32(1) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
              Left Join:  Filter: customer.c_custkey = __scalar_sq_1.o_custkey [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                  Projection: max(orders.o_custkey), orders.o_custkey, __always_true [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                    Aggregate: groupBy=[[orders.o_custkey, Boolean(true) AS __always_true]], aggr=[[max(orders.o_custkey)]] [o_custkey:Int64, __always_true:Boolean, max(orders.o_custkey):Int64;N]
                      TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    #[test]
    fn scalar_subquery_additional_filters_with_equal_clause() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .eq(col("orders.o_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(
                col("customer.c_custkey")
                    .eq(scalar_subquery(sq))
                    .and(col("c_custkey").eq(lit(1))),
            )?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey = __scalar_sq_1.max(orders.o_custkey) AND customer.c_custkey = Int32(1) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
              Left Join:  Filter: customer.c_custkey = __scalar_sq_1.o_custkey [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                  Projection: max(orders.o_custkey), orders.o_custkey, __always_true [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                    Aggregate: groupBy=[[orders.o_custkey, Boolean(true) AS __always_true]], aggr=[[max(orders.o_custkey)]] [o_custkey:Int64, __always_true:Boolean, max(orders.o_custkey):Int64;N]
                      TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test for correlated scalar subquery filter with disjunctions
    #[test]
    fn scalar_subquery_disjunction() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .eq(col("orders.o_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(
                col("customer.c_custkey")
                    .eq(scalar_subquery(sq))
                    .or(col("customer.c_custkey").eq(lit(1))),
            )?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey = __scalar_sq_1.max(orders.o_custkey) OR customer.c_custkey = Int32(1) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
              Left Join:  Filter: customer.c_custkey = __scalar_sq_1.o_custkey [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                  Projection: max(orders.o_custkey), orders.o_custkey, __always_true [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                    Aggregate: groupBy=[[orders.o_custkey, Boolean(true) AS __always_true]], aggr=[[max(orders.o_custkey)]] [o_custkey:Int64, __always_true:Boolean, max(orders.o_custkey):Int64;N]
                      TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    /// Test for correlated scalar subquery filter
    #[test]
    fn exists_subquery_correlated() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(test_table_scan_with_name("sq")?)
                .filter(out_ref_col(DataType::UInt32, "test.a").eq(col("sq.a")))?
                .aggregate(Vec::<Expr>::new(), vec![min(col("c"))])?
                .project(vec![min(col("c"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(test_table_scan_with_name("test")?)
            .filter(col("test.c").lt(scalar_subquery(sq)))?
            .project(vec![col("test.c")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: test.c [c:UInt32]
          Projection: test.a, test.b, test.c [a:UInt32, b:UInt32, c:UInt32]
            Filter: test.c < __scalar_sq_1.min(sq.c) [a:UInt32, b:UInt32, c:UInt32, min(sq.c):UInt32;N, a:UInt32;N, __always_true:Boolean;N]
              Left Join:  Filter: test.a = __scalar_sq_1.a [a:UInt32, b:UInt32, c:UInt32, min(sq.c):UInt32;N, a:UInt32;N, __always_true:Boolean;N]
                TableScan: test [a:UInt32, b:UInt32, c:UInt32]
                SubqueryAlias: __scalar_sq_1 [min(sq.c):UInt32;N, a:UInt32, __always_true:Boolean]
                  Projection: min(sq.c), sq.a, __always_true [min(sq.c):UInt32;N, a:UInt32, __always_true:Boolean]
                    Aggregate: groupBy=[[sq.a, Boolean(true) AS __always_true]], aggr=[[min(sq.c)]] [a:UInt32, __always_true:Boolean, min(sq.c):UInt32;N]
                      TableScan: sq [a:UInt32, b:UInt32, c:UInt32]
        "
        )
    }

    /// Test for non-correlated scalar subquery with no filters
    #[test]
    fn scalar_subquery_non_correlated_no_filters_with_non_equal_clause() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").lt(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey < __scalar_sq_1.max(orders.o_custkey) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N]
              Left Join:  Filter: Boolean(true) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N]
                  Projection: max(orders.o_custkey) [max(orders.o_custkey):Int64;N]
                    Aggregate: groupBy=[[]], aggr=[[max(orders.o_custkey)]] [max(orders.o_custkey):Int64;N]
                      TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    #[test]
    fn scalar_subquery_non_correlated_no_filters_with_equal_clause() -> Result<()> {
        let sq = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(col("customer.c_custkey").eq(scalar_subquery(sq)))?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey = __scalar_sq_1.max(orders.o_custkey) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N]
              Left Join:  Filter: Boolean(true) [c_custkey:Int64, c_name:Utf8, max(orders.o_custkey):Int64;N]
                TableScan: customer [c_custkey:Int64, c_name:Utf8]
                SubqueryAlias: __scalar_sq_1 [max(orders.o_custkey):Int64;N]
                  Projection: max(orders.o_custkey) [max(orders.o_custkey):Int64;N]
                    Aggregate: groupBy=[[]], aggr=[[max(orders.o_custkey)]] [max(orders.o_custkey):Int64;N]
                      TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    #[test]
    fn correlated_scalar_subquery_in_between_clause() -> Result<()> {
        let sq1 = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .eq(col("orders.o_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![min(col("orders.o_custkey"))])?
                .project(vec![min(col("orders.o_custkey"))])?
                .build()?,
        );
        let sq2 = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .filter(
                    out_ref_col(DataType::Int64, "customer.c_custkey")
                        .eq(col("orders.o_custkey")),
                )?
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let between_expr = Expr::Between(Between {
            expr: Box::new(col("customer.c_custkey")),
            negated: false,
            low: Box::new(scalar_subquery(sq1)),
            high: Box::new(scalar_subquery(sq2)),
        });

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(between_expr)?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey BETWEEN __scalar_sq_1.min(orders.o_custkey) AND __scalar_sq_2.max(orders.o_custkey) [c_custkey:Int64, c_name:Utf8, min(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
              Left Join:  Filter: customer.c_custkey = __scalar_sq_2.o_custkey [c_custkey:Int64, c_name:Utf8, min(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N, max(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
                Left Join:  Filter: customer.c_custkey = __scalar_sq_1.o_custkey [c_custkey:Int64, c_name:Utf8, min(orders.o_custkey):Int64;N, o_custkey:Int64;N, __always_true:Boolean;N]
                  TableScan: customer [c_custkey:Int64, c_name:Utf8]
                  SubqueryAlias: __scalar_sq_1 [min(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                    Projection: min(orders.o_custkey), orders.o_custkey, __always_true [min(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                      Aggregate: groupBy=[[orders.o_custkey, Boolean(true) AS __always_true]], aggr=[[min(orders.o_custkey)]] [o_custkey:Int64, __always_true:Boolean, min(orders.o_custkey):Int64;N]
                        TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                SubqueryAlias: __scalar_sq_2 [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                  Projection: max(orders.o_custkey), orders.o_custkey, __always_true [max(orders.o_custkey):Int64;N, o_custkey:Int64, __always_true:Boolean]
                    Aggregate: groupBy=[[orders.o_custkey, Boolean(true) AS __always_true]], aggr=[[max(orders.o_custkey)]] [o_custkey:Int64, __always_true:Boolean, max(orders.o_custkey):Int64;N]
                      TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }

    #[test]
    fn uncorrelated_scalar_subquery_in_between_clause() -> Result<()> {
        let sq1 = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .aggregate(Vec::<Expr>::new(), vec![min(col("orders.o_custkey"))])?
                .project(vec![min(col("orders.o_custkey"))])?
                .build()?,
        );
        let sq2 = Arc::new(
            LogicalPlanBuilder::from(scan_tpch_table("orders"))
                .aggregate(Vec::<Expr>::new(), vec![max(col("orders.o_custkey"))])?
                .project(vec![max(col("orders.o_custkey"))])?
                .build()?,
        );

        let between_expr = Expr::Between(Between {
            expr: Box::new(col("customer.c_custkey")),
            negated: false,
            low: Box::new(scalar_subquery(sq1)),
            high: Box::new(scalar_subquery(sq2)),
        });

        let plan = LogicalPlanBuilder::from(scan_tpch_table("customer"))
            .filter(between_expr)?
            .project(vec![col("customer.c_custkey")])?
            .build()?;

        assert_optimized_plan_equal!(
            plan,
            @r"
        Projection: customer.c_custkey [c_custkey:Int64]
          Projection: customer.c_custkey, customer.c_name [c_custkey:Int64, c_name:Utf8]
            Filter: customer.c_custkey BETWEEN __scalar_sq_1.min(orders.o_custkey) AND __scalar_sq_2.max(orders.o_custkey) [c_custkey:Int64, c_name:Utf8, min(orders.o_custkey):Int64;N, max(orders.o_custkey):Int64;N]
              Left Join:  Filter: Boolean(true) [c_custkey:Int64, c_name:Utf8, min(orders.o_custkey):Int64;N, max(orders.o_custkey):Int64;N]
                Left Join:  Filter: Boolean(true) [c_custkey:Int64, c_name:Utf8, min(orders.o_custkey):Int64;N]
                  TableScan: customer [c_custkey:Int64, c_name:Utf8]
                  SubqueryAlias: __scalar_sq_1 [min(orders.o_custkey):Int64;N]
                    Projection: min(orders.o_custkey) [min(orders.o_custkey):Int64;N]
                      Aggregate: groupBy=[[]], aggr=[[min(orders.o_custkey)]] [min(orders.o_custkey):Int64;N]
                        TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
                SubqueryAlias: __scalar_sq_2 [max(orders.o_custkey):Int64;N]
                  Projection: max(orders.o_custkey) [max(orders.o_custkey):Int64;N]
                    Aggregate: groupBy=[[]], aggr=[[max(orders.o_custkey)]] [max(orders.o_custkey):Int64;N]
                      TableScan: orders [o_orderkey:Int64, o_custkey:Int64, o_orderstatus:Utf8, o_totalprice:Float64;N]
        "
        )
    }
}
