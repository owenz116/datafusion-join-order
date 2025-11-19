use std::collections::HashMap;
use std::sync::Arc;

use datafusion_common::{Column, DataFusionError, Result as DFResult, TableReference};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_expr::logical_plan::{Join, LogicalPlan, SubqueryAlias, TableScan};
use datafusion_expr::{Expr, JoinConstraint, JoinType};

/// A base relation participating in a join island.
///
/// For Commit 1 we treat *any* leaf logical plan (typically
/// `TableScan` or `SubqueryAlias`) as an atom.
#[derive(Debug, Clone)]
pub struct JoinRelation {
    /// Stable index of this relation in the `JoinGraph.relations` vector.
    pub id: usize,
    /// A human-readable identifier (typically table or alias name).
    pub name: String,
    /// The full logical subplan for this relation (TableScan, SubqueryAlias, etc.).
    pub plan: LogicalPlan,
    /// Local predicates that can be pushed down to this relation.
    ///
    /// For Commit 1 this will be empty; we’ll populate it later when we
    /// start pulling apart Filters.
    pub filters: Vec<Expr>,
}

/// A binary equi-join edge between two relations in the join graph.
///
/// Moerkotte/Neumann generalize this to hyperedges and arbitrary
/// predicates; right now we restrict to `col = col` pairs.
#[derive(Debug, Clone)]
pub struct JoinEdge {
    /// Index into `JoinGraph.relations`
    pub left: usize,
    /// Index into `JoinGraph.relations`
    pub right: usize,
    /// The ON clause pairs for this edge. Each (left_expr, right_expr)
    /// is an equijoin `left_expr = right_expr`.
    pub on: Vec<(Expr, Expr)>,
}

/// A join island expressed as a graph over base relations.
#[derive(Debug, Clone)]
pub struct JoinGraph {
    pub relations: Vec<JoinRelation>,
    pub edges: Vec<JoinEdge>,
}

/// High-level entry point: given a logical plan node that is either
/// - a single INNER equi-join, possibly nested, or
/// - a leaf (no joins),
/// try to extract a JoinGraph.
///
/// Returns:
/// - Ok(Some(graph)) for a reorderable join island
/// - Ok(None) if the plan is *not* a pure inner equi-join island
///   (outer joins, non-equi filters, etc.)
pub fn extract_join_graph(root: &LogicalPlan) -> DFResult<Option<JoinGraph>> {
    let mut builder = GraphBuilder::default();
    if !builder.walk(root)? {
        return Ok(None);
    }

    let graph = JoinGraph {
        relations: builder.relations,
        edges: builder.edges,
    };

    if graph.relations.len() >= 2 && !graph.edges.is_empty() {
        Ok(Some(graph))
    } else {
        Ok(None)
    }
}

/// Internal helper for building a JoinGraph.
#[derive(Default)]
struct GraphBuilder {
    relations: Vec<JoinRelation>,
    edges: Vec<JoinEdge>,
    /// Map from relation (table / alias) to `relations` index.
    name_to_id: HashMap<TableReference, usize>,
    /// Did we see only “safe” inner equi-joins?
    ok: bool,
}

impl GraphBuilder {
    /// Recursively walk the plan. Returns `Ok(true)` if this subtree
    /// is entirely made of inner equi-joins + base relations.
    fn walk(&mut self, plan: &LogicalPlan) -> DFResult<bool> {
        match plan {
            LogicalPlan::Join(join) => self.handle_join(join),
            // We treat everything else as an atomic base relation *if and only if*
            // there are no joins inside it. This is conservative but safe.
            other => {
                if contains_join(other)? {
                    // We don’t (yet) support islands with subqueries containing joins.
                    self.ok = false;
                    Ok(false)
                } else {
                    self.add_base_relation(other);
                    Ok(true)
                }
            }
        }
    }

    fn handle_join(&mut self, join: &Join) -> DFResult<bool> {
        if !is_reorderable_inner_equi_join(join) {
            self.ok = false;
            return Ok(false);
        }

        // Recursively walk children. If either contains unsupported stuff, bail.
        let left_ok = self.walk(&join.left)?;
        let right_ok = self.walk(&join.right)?;
        if !left_ok || !right_ok {
            self.ok = false;
            return Ok(false);
        }

        // At this point we have populated `relations` and `name_to_id`
        // with all base relations under this join. Now map ON-clause
        // column pairs to relation indices.
        //
        // For Commit 1 we assume each side is a simple Column with a
        // non-empty `relation` field that identifies the base.
        let mut edge_pairs: Vec<(Expr, Expr)> = Vec::new();
        let mut left_rel_id: Option<usize> = None;
        let mut right_rel_id: Option<usize> = None;

        for (l_expr, r_expr) in &join.on {
            let (l_col, r_col) = match (as_column(l_expr), as_column(r_expr)) {
                (Some(lc), Some(rc)) => (lc, rc),
                _ => {
                    // non-column or expression join keys => not supported in this phase
                    self.ok = false;
                    return Ok(false);
                }
            };

            let l_ref = match &l_col.relation {
                Some(r) => r,
                None => {
                    self.ok = false;
                    return Ok(false);
                }
            };
            let r_ref = match &r_col.relation {
                Some(r) => r,
                None => {
                    self.ok = false;
                    return Ok(false);
                }
            };

            let l_id = self.lookup_relation_id(l_ref)?;
            let r_id = self.lookup_relation_id(r_ref)?;

            // Record that this edge connects l_id and r_id. All pairs in this join
            // share the same endpoints in the simple equi-join case.
            left_rel_id.get_or_insert(l_id);
            right_rel_id.get_or_insert(r_id);

            // Sanity: if multiple pairs disagree on which relations they connect, bail.
            if left_rel_id != Some(l_id) || right_rel_id != Some(r_id) {
                self.ok = false;
                return Ok(false);
            }

            edge_pairs.push((l_expr.clone(), r_expr.clone()));
        }

        if edge_pairs.is_empty() {
            // Should not happen for a valid equi-join, but be defensive.
            self.ok = false;
            return Ok(false);
        }

        self.edges.push(JoinEdge {
            left: left_rel_id.expect("left_rel_id must be set"),
            right: right_rel_id.expect("right_rel_id must be set"),
            on: edge_pairs,
        });

        Ok(true)
    }

    fn add_base_relation(&mut self, plan: &LogicalPlan) {
        let (table_ref, display_name) =
            relation_ref_and_name(plan).unwrap_or_else(|| (TableReference::bare("<anon>"), "<anon>".to_string()));

        // If we’ve already seen this base relation in this island, reuse its id.
        if let Some(&id) = self.name_to_id.get(&table_ref) {
            let _ = id;
            return;
        }

        let id = self.relations.len();
        self.name_to_id.insert(table_ref, id);

        self.relations.push(JoinRelation {
            id,
            name: display_name,
            plan: plan.clone(),
            filters: Vec::new(),
        });
    }

    fn lookup_relation_id(&self, tref: &TableReference) -> DFResult<usize> {
        self.name_to_id
            .get(tref)
            .copied()
            .ok_or_else(|| DataFusionError::Plan(format!(
                "Moerkotte join graph: unknown relation in join key: {tref}"
            )))
    }
}

/// Very conservative check: is this join something we are willing to
/// reorder in Commit 1?
fn is_reorderable_inner_equi_join(join: &Join) -> bool {
    if join.join_type != JoinType::Inner {
        return false;
    }
    if join.join_constraint != JoinConstraint::On {
        return false;
    }
    if join.filter.is_some() {
        // We ignore non-equi predicates for now
        return false;
    }

    // ON must be all column = column; we validate this more strictly in handle_join.
    !join.on.is_empty()
}

/// Check whether a plan subtree contains any Join nodes at all.
fn contains_join(plan: &LogicalPlan) -> DFResult<bool> {
    let mut found = false;

    plan.apply(|node| {
        if matches!(node, LogicalPlan::Join(_)) {
            found = true;
            Ok(TreeNodeRecursion::Stop)
        } else {
            Ok(TreeNodeRecursion::Continue)
        }
    })?;

    Ok(found)
}

fn as_column(expr: &Expr) -> Option<&Column> {
    if let Expr::Column(c) = expr {
        Some(c)
    } else {
        None
    }
}

/// Try to derive a `TableReference` and a human-readable name for a base relation.
///
/// For Commit 1:
/// - SubqueryAlias → alias (as TableReference + String)
/// - TableScan → table_name (TableReference + its Display)
fn relation_ref_and_name(plan: &LogicalPlan) -> Option<(TableReference, String)> {
    match plan {
        LogicalPlan::SubqueryAlias(SubqueryAlias { alias, .. }) => {
            // `alias` is already a TableReference
            let tref = alias.clone();
            Some((tref.clone(), tref.to_string()))
        }
        LogicalPlan::TableScan(TableScan { table_name, .. }) => {
            Some((table_name.clone(), table_name.to_string()))
        }
        _ => None,
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::Result as DFResult;
    use datafusion_expr::{
        logical_plan::LogicalPlanBuilder,
        Expr,
        JoinType,
        LogicalPlan,
        TableSource,
        TableProviderFilterPushDown,
    };

    #[derive(Debug)]
    struct RawTableSource;

    impl TableSource for RawTableSource {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn schema(&self) -> arrow::datatypes::SchemaRef {
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
            ]))
        }

        fn supports_filters_pushdown(
            &self,
            filters: &[&Expr],
        ) -> datafusion_common::Result<Vec<TableProviderFilterPushDown>> {
            Ok(vec![TableProviderFilterPushDown::Inexact])
        }
    }

    fn dummy_scan(name: &str) -> LogicalPlan {
        LogicalPlanBuilder::scan(
            name.to_string(),
            Arc::new(RawTableSource),
            None,
        )
            .unwrap()
            .build()
            .unwrap()
    }

    #[test]
    fn extract_simple_two_way_join_graph() {
        let a_scan = dummy_scan("a");
        let b_scan = dummy_scan("b");

        // Simple left-deep join:
        //
        //   Join (a.id = b.id)
        //   /   \
        //   a   b
        let ab_join = LogicalPlanBuilder::from(a_scan)
            .join(
                b_scan,
                JoinType::Inner,
                (vec!["id"], vec!["id"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        let graph = extract_join_graph(&ab_join)
            .expect("ok")
            .expect("some join graph");

        assert_eq!(graph.relations.len(), 2);
        assert_eq!(graph.edges.len(), 1);

        let names: Vec<_> = graph.relations.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"a"));
        assert!(names.contains(&"b"));
    }
}
