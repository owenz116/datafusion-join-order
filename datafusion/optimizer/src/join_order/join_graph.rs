use std::collections::HashMap;
use std::sync::Arc;

use datafusion_common::{Column, DataFusionError, Result as DFResult, TableReference};
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion, Transformed};
use datafusion_expr::logical_plan::{
    Join, LogicalPlan, SubqueryAlias, TableScan, LogicalPlanBuilder,
};
use datafusion_expr::{Expr, JoinConstraint, JoinType};
use datafusion_expr::utils::{conjunction, split_conjunction};



type BitSet = u64;


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

/// A join predicate hyperedge over one or more relations.
///
/// In the Moerkotte/Neumann terminology, this is the query hypergraph
/// edge: all relations whose columns appear in the equality predicate.
#[derive(Debug, Clone)]
pub struct JoinPredicate {
    /// Bitset of relations (indices into `JoinGraph.relations`) that
    /// this predicate references.
    pub rel_mask: BitSet,
    /// The full equality expression, e.g. `a.x + b.y = c.z + d.w`.
    ///
    /// For simple column = column joins this will just be `a.x = b.y`.
    pub expr: Expr,
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

/// A join island expressed as a graph / hypergraph over base relations.
#[derive(Debug, Clone)]
pub struct JoinGraph {
    pub relations: Vec<JoinRelation>,
    /// Pairwise edges used mainly for “classic” 2-way equijoins.
    pub edges: Vec<JoinEdge>,
    /// Hyperedges over arbitrary sets of relations, one per equality
    /// predicate in the logical join tree.
    pub predicates: Vec<JoinPredicate>,
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
        predicates: builder.predicates,
    };
    
    if graph.relations.len() >= 2 && (!graph.edges.is_empty() || !graph.predicates.is_empty()) {
        Ok(Some(graph))
    } else {
        Ok(None)
    }

}


#[derive(Default)]
struct GraphBuilder {
    relations: Vec<JoinRelation>,
    edges: Vec<JoinEdge>,
    /// All join predicates as hyperedges.
    predicates: Vec<JoinPredicate>,
    /// Map from relation (table / alias) to `relations` index.
    name_to_id: HashMap<TableReference, usize>,
    /// Did we see only “safe” inner equi-joins?
    ok: bool,
}

impl GraphBuilder {
    /// Collect the set of relation-ids referenced by `expr` using
    /// Expr::column_refs(), and (optionally) return a single relation-id
    /// if all columns in `expr` come from the same relation.
    fn rel_mask_and_single_id_for_expr(
        &self,
        expr: &Expr,
    ) -> DFResult<(BitSet, Option<usize>)> {
        use std::collections::HashSet;

        let cols: HashSet<&Column> = expr.column_refs();

        let mut mask: BitSet = 0;
        let mut single: Option<usize> = None;

        for col in cols {
            let tref = match &col.relation {
                Some(r) => r,
                None => {
                    // We require qualified columns inside join keys so we can
                    // map them to base relations in this island.
                    return Err(DataFusionError::Plan(format!(
                        "Moerkotte join graph: unqualified column {} in join expression",
                        col.name
                    )));
                }
            };

            let rel_id = self.lookup_relation_id(tref)?;
            let bit = 1u64 << rel_id;
            if mask & bit == 0 {
                mask |= bit;

                match single {
                    None => single = Some(rel_id),
                    Some(prev) if prev == rel_id => {
                        // still only one relation so far
                    }
                    Some(_) => {
                        // expression references multiple relations
                        single = None;
                    }
                }
            }
        }

        Ok((mask, single))
    }

    /// Add (or extend) a simple 2-way JoinEdge between `left` and `right`
    /// for a single ON pair (l_expr, r_expr).
    fn add_binary_edge(&mut self, left: usize, right: usize, l_expr: Expr, r_expr: Expr) {
        let (a, b) = if left <= right { (left, right) } else { (right, left) };
        if let Some(edge) = self
            .edges
            .iter_mut()
            .find(|e| (e.left == a && e.right == b) || (e.left == b && e.right == a))
        {
            edge.on.push((l_expr, r_expr));
        } else {
            self.edges.push(JoinEdge {
                left: a,
                right: b,
                on: vec![(l_expr, r_expr)],
            });
        }
    }
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
        // with all base relations under this join. Now map each ON-clause
        // equality to:
        //   * a JoinPredicate hyperedge over all referenced relations
        //   * optionally, a 2-way JoinEdge if each side touches exactly
        //     one relation and they differ.
        for (l_expr, r_expr) in &join.on {
            // Build hyperedge: union of all relation ids in l_expr and r_expr.
            let (left_mask, left_single) =
                match self.rel_mask_and_single_id_for_expr(l_expr) {
                    Ok(v) => v,
                    Err(_) => {
                        self.ok = false;
                        return Ok(false);
                    }
                };
            let (right_mask, right_single) =
                match self.rel_mask_and_single_id_for_expr(r_expr) {
                    Ok(v) => v,
                    Err(_) => {
                        self.ok = false;
                        return Ok(false);
                    }
                };

            let rel_mask = left_mask | right_mask;
            if rel_mask == 0 {
                // Should not happen for a valid equi-join, but be defensive.
                self.ok = false;
                return Ok(false);
            }

            // Record hyperedge for the full equality.
            self.predicates.push(JoinPredicate {
                rel_mask,
                expr: l_expr.clone().eq(r_expr.clone()),
            });

            // If this ON pair is "simple" (each side references exactly one
            // relation, and they are different), keep a binary edge as well.
            if let (Some(l_id), Some(r_id)) = (left_single, right_single) {
                if l_id != r_id {
                    self.add_binary_edge(l_id, r_id, l_expr.clone(), r_expr.clone());
                } else {
                    // Equality within the same relation (e.g. self-join key),
                    // not useful as a join edge here.
                }
            } else {
                // True hyperedge like a.x + b.y = c.z + d.w: we only keep it
                // in `predicates` (hypergraph), no simple JoinEdge.
            }
        }

        // treat filter predicates as hyperedges too
        if let Some(filter_expr) = &join.filter {
            for conjunct in split_conjunction(filter_expr) {
                // Compute the set of relations this predicate touches.
                let (rel_mask, _maybe_single) =
                    match self.rel_mask_and_single_id_for_expr(conjunct) {
                        Ok(v) => v,
                        Err(_) => {
                            // If we can't map the expression cleanly to our relation ids,
                            // bail out for now: this join tree is not (yet) reorderable.
                            self.ok = false;
                            return Ok(false);
                        }
                    };

                // Predicates that only touch 0 or 1 relation are "local" filters;
                // they don't participate in join connectivity.
                if rel_mask.count_ones() < 2 {
                    continue;
                }

                // Record a hyperedge for this filter predicate. Unlike ON pairs,
                // we don't try to synthesize a binary JoinEdge: this might be a
                // three- or four-way predicate.
                self.predicates.push(JoinPredicate {
                    rel_mask,
                    expr: conjunct.clone(),
                });
            }
        }

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

fn is_reorderable_inner_equi_join(join: &Join) -> bool {
    // For now we only reorder INNER ... ON joins.
    if join.join_type != JoinType::Inner {
        return false;
    }
    if join.join_constraint != JoinConstraint::On {
        return false;
    }

    // We now allow ON with or without equi-pairs, and arbitrary filter predicates.
    true
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

/// DP entry for a subset of relations (bitset) in the DPhyp-style table.
#[derive(Debug, Clone)]
pub struct DPhypPlanEntry {
    /// Bitset encoding the subset of relations this entry refers to.
    pub subset: BitSet,
    /// If this subset was obtained by joining `left` and `right`,
    /// these fields store the two child subsets. `None` for base relations.
    pub left: Option<BitSet>,
    pub right: Option<BitSet>,
    /// Accumulated cost of the best plan for this subset.
    pub cost: f64,
    /// Estimated cardinality for this subset (placeholder model for now).
    pub cardinality: f64,
}

/// Result of running the DPhyp-style DP on a `JoinGraph`.
#[derive(Debug, Clone)]
pub struct DPhypResult {
    /// Best plan for each connected subset (keyed by subset bitmask).
    pub plans: HashMap<BitSet, DPhypPlanEntry>,
    /// Bitmask of the full connected set of relations we optimized for.
    pub full_subset: BitSet,
}

/// Run a DPhyp-style DP over the given join graph.
///
/// - `base_cardinalities`: optional per-relation base cardinalities,
///   indexed by `JoinRelation.id` (0..n-1). If `None` or too short,
///   we fall back to 1.0 for that relation.
///
/// Placeholder cost model:
///   * base relation i: card_i = base_card_i, cost_i = card_i
///   * join of subsets L and R:
///       result_card = card(L) * card(R)
///       result_cost = cost(L) + cost(R) + result_card
///
/// This mirrors the classic System R–style "sum of intermediate costs"
/// idea and is easy to swap out later with a DataFusion stats-based model. 
pub fn dphyp_optimize_join_graph<M: DPhypCostModel>(
    graph: &JoinGraph,
    model: &M,
) -> Option<DPhypResult> {
    let n = graph.relations.len();
    if n == 0 {
        return None;
    }
    if n > 63 {
        return None;
    }

    let neighbors = build_neighbor_masks(graph);
    let full_mask: BitSet = if n == 64 {
        u64::MAX
    } else {
        (1u64 << n) - 1
    };

    let mut connected_subsets: Vec<BitSet> = Vec::new();
    for mask in 1..=full_mask {
        if is_connected_mask(mask, &neighbors) {
            connected_subsets.push(mask);
        }
    }
    connected_subsets.sort_by_key(|m| bit_count(*m));

    let mut plans: HashMap<BitSet, DPhypPlanEntry> = HashMap::new();

    // ---- base relations (singletons) ----
    for rel in 0..n {
        let mask = 1u64 << rel;

        let (base_cost, base_card) = model.base_cost_cardinality(rel, graph);
        plans.insert(
            mask,
            DPhypPlanEntry {
                subset: mask,
                left: None,
                right: None,
                cost: base_cost,
                cardinality: base_card,
            },
        );
    }

    // ---- larger connected subsets ----
    for &subset in &connected_subsets {
        if bit_count(subset) <= 1 {
            continue;
        }

        let mut best: Option<DPhypPlanEntry> = None;

        let mut sub = subset & (subset - 1);
        while sub > 0 {
            let left_mask = sub;
            let right_mask = subset ^ left_mask;
            if right_mask == 0 {
                sub = (sub - 1) & subset;
                continue;
            }

            let left_plan = match plans.get(&left_mask) {
                Some(p) => p,
                None => {
                    sub = (sub - 1) & subset;
                    continue;
                }
            };
            let right_plan = match plans.get(&right_mask) {
                Some(p) => p,
                None => {
                    sub = (sub - 1) & subset;
                    continue;
                }
            };

            if !has_cross_edge(left_mask, right_mask, graph) {
                sub = (sub - 1) & subset;
                continue;
            }

            // delegate completely to cost model
            let (join_cost, result_card) =
                model.join_cost_cardinality(left_plan, right_plan, graph);

            let result_cost = left_plan.cost + right_plan.cost + join_cost;

            match &mut best {
                None => {
                    best = Some(DPhypPlanEntry {
                        subset,
                        left: Some(left_mask),
                        right: Some(right_mask),
                        cost: result_cost,
                        cardinality: result_card,
                    });
                }
                Some(b) => {
                    if result_cost < b.cost {
                        *b = DPhypPlanEntry {
                            subset,
                            left: Some(left_mask),
                            right: Some(right_mask),
                            cost: result_cost,
                            cardinality: result_card,
                        };
                    }
                }
            }

            sub = (sub - 1) & subset;
        }

        if let Some(entry) = best {
            plans.insert(subset, entry);
        }
    }

    if !plans.contains_key(&full_mask) {
        return None;
    }

    Some(DPhypResult {
        plans,
        full_subset: full_mask,
    })
}
/// Build bitset neighbor masks for each relation in the join hypergraph.
///
/// neighbors[i] has bits set for all j such that there exists at least one
/// join predicate whose `rel_mask` contains *both* i and j.
/// Build bitset neighbor masks for each relation in the join hypergraph.
///
/// neighbors[i] has bits set for all j such that there exists at least one
/// join predicate or binary join edge whose support contains *both* i and j.
fn build_neighbor_masks(graph: &JoinGraph) -> Vec<BitSet> {
    let n = graph.relations.len();
    let mut neighbors = vec![0u64; n];

    // 1) Hyperedges: project each predicate's rel_mask to a clique
    for pred in &graph.predicates {
        let mut rels = pred.rel_mask;

        while rels != 0 {
            let i = rels.trailing_zeros() as usize;
            let bit_i = 1u64 << i;
            rels ^= bit_i;

            let others = pred.rel_mask ^ bit_i;
            neighbors[i] |= others;
        }
    }

    // 2) Simple edges: also connect left/right directly
    for edge in &graph.edges {
        if edge.left < n && edge.right < n {
            let l_bit = 1u64 << edge.left;
            let r_bit = 1u64 << edge.right;
            neighbors[edge.left] |= r_bit;
            neighbors[edge.right] |= l_bit;
        }
    }

    neighbors
}


/// Count bits in a BitSet.
fn bit_count(mask: BitSet) -> usize {
    mask.count_ones() as usize
}

/// Check if `mask` induces a connected subgraph in the query graph
/// represented by `neighbors`.
fn is_connected_mask(mask: BitSet, neighbors: &[BitSet]) -> bool {
    if mask == 0 {
        return false;
    }

    // Start BFS from the lowest-numbered relation in `mask`.
    let start = mask.trailing_zeros() as usize;
    let mut visited: BitSet = 0;
    let mut stack: Vec<usize> = Vec::new();

    visited |= 1u64 << start;
    stack.push(start);

    while let Some(v) = stack.pop() {
        let mut nbrs = neighbors[v] & mask;
        while nbrs != 0 {
            let idx = nbrs.trailing_zeros() as usize;
            let bit = 1u64 << idx;
            nbrs ^= bit;
            if visited & bit == 0 {
                visited |= bit;
                stack.push(idx);
            }
        }
    }

    visited == mask
}
/// Check whether there is any join condition crossing the cut (A, B).
///
/// We treat both:
///   - hyperedges in `graph.predicates` (support sets with > 1 rel)
///   - simple binary edges in `graph.edges`
///
/// as evidence that the cut is "joinable" (i.e., not a pure cross product).
fn has_cross_edge(a: BitSet, b: BitSet, graph: &JoinGraph) -> bool {
    // 1) Hyperedges: any predicate whose support intersects both A and B
    for pred in &graph.predicates {
        let mask = pred.rel_mask;
        if (mask & a) != 0 && (mask & b) != 0 {
            return true;
        }
    }

    // 2) Simple binary edges: any edge connecting a relation in A to one in B
    for edge in &graph.edges {
        let left_bit = 1u64 << edge.left;
        let right_bit = 1u64 << edge.right;

        let crosses =
            (left_bit & a != 0 && right_bit & b != 0) ||
                (left_bit & b != 0 && right_bit & a != 0);

        if crosses {
            return true;
        }
    }

    false
}


/// Rebuild the best join plan (LogicalPlan) for a join island
/// described by `graph`, using the DP table in `dp`.
///
/// This uses the `full_subset` bitmask as the island root.
pub fn build_best_join_plan_from_dphyp(
    graph: &JoinGraph,
    dp: &DPhypResult,
) -> DFResult<LogicalPlan> {
    build_plan_for_subset(dp.full_subset, graph, dp)
}

/// Recursively rebuild a plan for a subset:
///  - If subset is a singleton → return the base relation LogicalPlan
///  - Else → recurse into left/right subsets and join them with the
///    predicates crossing the cut.
fn build_plan_for_subset(
    subset: BitSet,
    graph: &JoinGraph,
    dp: &DPhypResult,
) -> DFResult<LogicalPlan> {
    let entry = dp
        .plans
        .get(&subset)
        .ok_or_else(|| DataFusionError::Internal(format!(
            "DPhyp reconstruction: no DP entry for subset {subset:b}"
        )))?;

    match (entry.left, entry.right) {
        (Some(left_subset), Some(right_subset)) => {
            // Non-leaf: recursively build children and then a Join
            let left_plan = build_plan_for_subset(left_subset, graph, dp)?;
            let right_plan = build_plan_for_subset(right_subset, graph, dp)?;

            build_join_for_children(left_subset, &left_plan, right_subset, &right_plan, graph)
        }
        (None, None) => {
            // Leaf: subset should contain exactly one relation bit
            let rel_idx = subset.trailing_zeros() as usize;

            if rel_idx >= graph.relations.len() {
                return Err(DataFusionError::Internal(format!(
                    "DPhyp reconstruction: subset {subset:b} refers to relation {rel_idx}, \
                     but graph only has {} relations",
                    graph.relations.len()
                )));
            }

            Ok(graph.relations[rel_idx].plan.clone())
        }
        _ => Err(DataFusionError::Internal(format!(
            "DPhyp reconstruction: malformed DP entry for subset {subset:b} \
             (expected either both children or none)"
        ))),
    }
}

/// Build a filter expression for the join node that joins `left_subset` and
/// `right_subset`.
///
/// For each JoinPredicate `p` in the hypergraph, we attach `p.expr` at the
/// *lowest* join node whose subset contains all relations in `p.rel_mask`
/// and where `p` crosses the cut (i.e. references at least one relation on
/// each side).
///
/// This mirrors DuckDB's behavior: simple 2-way equi-joins become join keys,
/// while more complex equalities over multiple relations are kept as filters.
fn build_filter_for_cut(
    left_subset: BitSet,
    right_subset: BitSet,
    graph: &JoinGraph,
) -> Option<Expr> {
    let union = left_subset | right_subset;
    let mut exprs = Vec::new();

    for pred in &graph.predicates {
        let mask = pred.rel_mask;
        if mask == 0 {
            continue;
        }

        // Predicate must mention at least one relation from *each* side
        // of the cut, otherwise it is local to one child and will be
        // attached lower in the tree.
        if (mask & left_subset) == 0 || (mask & right_subset) == 0 {
            continue;
        }

        // All referenced relations must be available at this node.
        if (mask & union) != mask {
            continue;
        }

        exprs.push(pred.expr.clone());
    }

    // AND them together if there are any.
    conjunction(exprs)
}


/// Build a Join node from two child plans and their subsets.
///
/// This:
/// - collects all edges in the join graph that cross the (left_subset, right_subset) cut
/// - orients each pair so that `left_cols[i]` refers to the left child,
///   `right_cols[i]` refers to the right child
/// - collects all hyperedge predicates that become valid at this node
/// - builds a `LogicalPlan` via `LogicalPlanBuilder::join`
fn build_join_for_children(
    left_subset: BitSet,
    left_plan: &LogicalPlan,
    right_subset: BitSet,
    right_plan: &LogicalPlan,
    graph: &JoinGraph,
) -> DFResult<LogicalPlan> {
    // Classic equi-join keys from 2-way JoinEdge entries.
    let (left_cols, right_cols) = join_keys_for_cut(left_subset, right_subset, graph)?;

    // Hyperedge predicates that "cross" this cut and whose relation set
    // is fully contained in left_subset ∪ right_subset.
    let filter_expr = build_filter_for_cut(left_subset, right_subset, graph);

    let has_keys = !left_cols.is_empty();

    if !has_keys && filter_expr.is_none() {
        // DP should never produce such a split: there must be at least
        // one predicate hyperedge crossing the cut, otherwise the graph
        // is disconnected. Treat as a safety check.
        return Err(DataFusionError::Plan(
            "DPhyp reconstruction: no join predicates across cut".to_string(),
        ));
    }

    // Case 1: we have equi-join keys (possibly plus extra filters).
    if has_keys {
        return LogicalPlanBuilder::from(left_plan.clone())
            .join(
                right_plan.clone(),
                JoinType::Inner, // DPhyp currently only handles inner joins
                (left_cols, right_cols),
                filter_expr,
            )?
            .build();
    }

    // Case 2: no binary equi-join keys, but there *is* a hyperedge that
    // crosses the cut. Build a filter-only join (e.g. nested loop).
    LogicalPlanBuilder::from(left_plan.clone())
        .join(
            right_plan.clone(),
            JoinType::Inner,
            // No key columns; planner will treat this as a non-equi join
            // with a join filter.
            (Vec::<Column>::new(), Vec::<Column>::new()),
            filter_expr,
        )?
        .build()
}


/// For a given cut (left_subset, right_subset), collect all ON-clause
/// columns from `graph.edges` that connect a relation in left_subset
/// with a relation in right_subset.
///
/// We must be careful about orientation: each JoinEdge is stored with
/// endpoints (edge.left, edge.right) and `edge.on` is a Vec<(Expr, Expr)>
/// with that ordering. But in the DP plan, either relation might end up
/// in the left or the right subset. So we may need to flip the pair.
fn join_keys_for_cut(
    left_subset: BitSet,
    right_subset: BitSet,
    graph: &JoinGraph,
) -> DFResult<(Vec<Column>, Vec<Column>)> {
    let mut left_cols = Vec::new();
    let mut right_cols = Vec::new();

    for edge in &graph.edges {
        let left_bit = 1u64 << edge.left;
        let right_bit = 1u64 << edge.right;

        let left_in_left = (left_subset & left_bit) != 0;
        let left_in_right = (right_subset & left_bit) != 0;
        let right_in_left = (left_subset & right_bit) != 0;
        let right_in_right = (right_subset & right_bit) != 0;

        // Edge crosses the cut if one endpoint is on the left side and
        // the other is on the right side (in either orientation).
        let crosses_normal = left_in_left && right_in_right;
        let crosses_flipped = right_in_left && left_in_right;

        if !crosses_normal && !crosses_flipped {
            continue;
        }

        let flipped = crosses_flipped;

        for (l_expr, r_expr) in &edge.on {
            let (l_col, r_col) = match (as_column(l_expr), as_column(r_expr)) {
                (Some(lc), Some(rc)) => (lc, rc),
                _ => {
                    // Should not happen: Commit 1 only adds Expr::Column pairs
                    continue;
                }
            };

            if !flipped {
                left_cols.push(l_col.clone());
                right_cols.push(r_col.clone());
            } else {
                // Swap orientation so left_cols always refer to the left child.
                left_cols.push(r_col.clone());
                right_cols.push(l_col.clone());
            }
        }
    }

    Ok((left_cols, right_cols))
}



/// Cost model for DPhyp-style DP over a `JoinGraph`.
///
/// The DP recurrence is:
///   cost(S) = min over splits S = L ∪ R:
///       cost(L) + cost(R) + join_cost(L,R)
///
/// `base_cost_cardinality` and `join_cost_cardinality` let you
/// define both `cost` and `cardinality` in one place.
pub trait DPhypCostModel {
    /// Cost and cardinality of a base relation (singleton subset)
    /// identified by its relation id in `graph.relations`.
    fn base_cost_cardinality(
        &self,
        rel_id: usize,
        graph: &JoinGraph,
    ) -> (f64 /*cost*/, f64 /*cardinality*/);

    /// Incremental join cost and resulting cardinality for joining
    /// `left` and `right` subsets.
    ///
    /// Returned `(join_cost, result_cardinality)` is interpreted as:
    ///
    ///   cost(S) = cost(L) + cost(R) + join_cost
    ///   card(S) = result_cardinality
    fn join_cost_cardinality(
        &self,
        left: &DPhypPlanEntry,
        right: &DPhypPlanEntry,
        graph: &JoinGraph,
    ) -> (f64 /*join_cost*/, f64 /*result_cardinality*/);
}

/// Simple placeholder cost model:
///
/// - base: cost = card = given base_cardinalities[i] or 1.0
/// - join: result_card = left.card * right.card
///         join_cost  = result_card
///
/// So:
///   cost(S) = cost(L) + cost(R) + card(S)
pub struct SimpleCardinalityCostModel<'a> {
    pub base_cardinalities: Option<&'a [f64]>,
}

/// Cost model that incorporates per-predicate selectivity.
///
/// Cardinality model:
///   card(S) = (∏ base_card(i) for i in S)
///             * (∏ sel(p) for all predicates p with rel_mask(p) ⊆ S)
///
/// where:
///   base_card(i)          comes from `base_cardinalities` or defaults to 1.0
///   sel(p) for predicate `p` comes from `predicate_selectivities[p_index]`
///   or defaults to 1.0 if not provided.
///
/// We deliberately compute `card(S)` from scratch for each subset `S` to
/// avoid double-counting predicate selectivities across different join
/// trees: every DP entry for the same subset mask gets the same cardinality.
pub struct PredicateSelectivityCostModel<'a> {
    /// Optional per-relation base cardinalities indexed by relation id.
    pub base_cardinalities: Option<&'a [f64]>,
    /// Optional per-predicate selectivities indexed by `graph.predicates`
    /// order. If shorter than `graph.predicates.len()`, missing entries
    /// default to 1.0.
    pub predicate_selectivities: Option<&'a [f64]>,
}

impl<'a> PredicateSelectivityCostModel<'a> {
    fn base_card(&self, rel_id: usize) -> f64 {
        self.base_cardinalities
            .and_then(|v| v.get(rel_id).copied())
            .unwrap_or(1.0)
    }

    fn predicate_sel(&self, pred_idx: usize) -> f64 {
        self.predicate_selectivities
            .and_then(|v| v.get(pred_idx).copied())
            .unwrap_or(1.0)
    }

    /// Compute cardinality for a subset mask from scratch:
    ///   card(S) = ∏ base_card(i) * ∏ sel(p)
    /// where `p.rel_mask ⊆ S`.
    fn subset_cardinality(&self, subset: BitSet, graph: &JoinGraph) -> f64 {
        let mut card = 1.0_f64;

        // Multiply base cardinalities for all relations in `subset`.
        let mut mask = subset;
        while mask != 0 {
            let i = mask.trailing_zeros() as usize;
            mask ^= 1u64 << i;
            card *= self.base_card(i);
        }

        // Multiply selectivities for all predicates whose support is
        // fully contained within `subset`.
        for (idx, pred) in graph.predicates.iter().enumerate() {
            let rel_mask = pred.rel_mask;
            if rel_mask != 0 && (rel_mask & subset) == rel_mask {
                card *= self.predicate_sel(idx);
            }
        }

        card
    }
}

impl<'a> DPhypCostModel for PredicateSelectivityCostModel<'a> {
    fn base_cost_cardinality(
        &self,
        rel_id: usize,
        _graph: &JoinGraph,
    ) -> (f64, f64) {
        let card = self.base_card(rel_id);
        // For now we mirror System R-style: base cost = base cardinality.
        (card, card)
    }

    fn join_cost_cardinality(
        &self,
        left: &DPhypPlanEntry,
        right: &DPhypPlanEntry,
        graph: &JoinGraph,
    ) -> (f64, f64) {
        let subset = left.subset | right.subset;

        // Compute cardinality of the *result* subset from scratch using
        // base cards + predicate selectivities.
        let result_card = self.subset_cardinality(subset, graph);

        // Simple join-cost model: cost contribution of this join step is
        // proportional to the size of the result.
        let join_cost = result_card;

        (join_cost, result_card)
    }
}


impl<'a> DPhypCostModel for SimpleCardinalityCostModel<'a> {
    fn base_cost_cardinality(
        &self,
        rel_id: usize,
        _graph: &JoinGraph,
    ) -> (f64, f64) {
        let card = self
            .base_cardinalities
            .and_then(|v| v.get(rel_id).copied())
            .unwrap_or(1.0);
        (card, card)
    }

    fn join_cost_cardinality(
        &self,
        left: &DPhypPlanEntry,
        right: &DPhypPlanEntry,
        _graph: &JoinGraph,
    ) -> (f64, f64) {
        let result_card = left.cardinality * right.cardinality;
        let join_cost = result_card;
        (join_cost, result_card)
    }
}

/// Run DPhyp-based join order optimization recursively over a plan.
///
/// This walks the plan bottom-up using `TreeNode::transform_up`.
/// For every `LogicalPlan::Join` subtree that can be expressed as a
/// reorderable join island (`extract_join_graph` returns `Some(..)`),
/// we invoke `dphyp_optimize_join_graph` and reconstruct the best join
/// tree for that island.
///
/// Non-join operators act as *island boundaries*, but their children
/// are still visited recursively, so a shape like
///
///   Join(
///     Join(
///       a,
///       Filter(
///         Aggregate(   // <- boundary
///           D_island   // <- inner join island
///         )
///       )
///     ),
///     c,
///   )
///
/// will first optimize `D_island`, then leave the outer joins as-is
/// because they cross the `Filter/Aggregate` boundary.
pub fn optimize_joins_in_plan<M: DPhypCostModel>(
    plan: &LogicalPlan,
    model: &M,
) -> DFResult<LogicalPlan> {
    // `transform_up` rewrites children first, then the current node,
    // which is exactly what we want: optimize inner islands before
    // deciding what to do with outer joins.
    let transformed = plan.clone().transform_up(|node| {
        match node {
            LogicalPlan::Join(join) => {
                let join_plan = LogicalPlan::Join(join.clone());

                // Try to view this join subtree as a join island.
                match extract_join_graph(&join_plan)? {
                    Some(graph) => {
                        if let Some(dp) = dphyp_optimize_join_graph(&graph, model) {
                            let best = build_best_join_plan_from_dphyp(&graph, &dp)?;
                            Ok(Transformed::yes(best))
                        } else {
                            // Graph looked reorderable, but DP couldn't find
                            // a full plan (e.g. disconnected): leave as-is.
                            Ok(Transformed::no(join_plan))
                        }
                    }
                    None => {
                        // Not a pure inner-island according to our extractor
                        Ok(Transformed::no(join_plan))
                    }
                }
            }
            other => {
                // Non-join nodes are left as-is; their children have already
                // been optimized by the bottom-up traversal.
                Ok(Transformed::no(other))
            }
        }
    })?;

    Ok(transformed.data)
}




#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use datafusion_expr::logical_plan::Filter;
    use datafusion_expr::lit;


    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion_common::Spans;
    use datafusion_expr::{logical_plan::LogicalPlanBuilder, Expr, JoinType, LogicalPlan, TableSource, TableProviderFilterPushDown, BinaryExpr};

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

    #[test]
    fn dphyp_two_way_join_cost() {
        // Graph: A -- B
        let rel_a = JoinRelation {
            id: 0,
            name: "A".to_string(),
            plan: dummy_scan("A"),
            filters: vec![],
        };
        let rel_b = JoinRelation {
            id: 1,
            name: "B".to_string(),
            plan: dummy_scan("B"),
            filters: vec![],
        };

        let relations = vec![rel_a, rel_b];

        let edges = vec![
            // A <-> B
            JoinEdge {
                left: 0,
                right: 1,
                on: vec![],
            },
        ];
        
        let predicates = vec![];

        let graph = JoinGraph { relations, edges, predicates };

        // base cardinalities: |A| = 1000, |B| = 10
        let base = [1000.0_f64, 10.0_f64];

        let model = SimpleCardinalityCostModel {
            base_cardinalities: Some(&base),
        };

        let result =
            dphyp_optimize_join_graph(&graph, &model).expect("expected a DP result");


        // let result =
        //     dphyp_optimize_join_graph(&graph, Some(&base)).expect("expected a DP result");

        assert_eq!(result.full_subset, 0b11);

        let p1 = result.plans.get(&0b01).expect("plan for {A}");
        assert_eq!(p1.cardinality, 1000.0);
        let p2 = result.plans.get(&0b10).expect("plan for {B}");
        assert_eq!(p2.cardinality, 10.0);

        let p12 = result.plans.get(&0b11).expect("plan for {A,B}");
        assert_eq!(p12.cardinality, 1000.0 * 10.0);
        assert_eq!(p12.cost, 1000.0 + 10.0 + 1000.0 * 10.0);
        assert!(p12.left.is_some() && p12.right.is_some());
    }

    #[test]
    fn dphyp_three_way_chain_prefers_middle_relation_first() {
        // Graph: A -- B -- C
        //
        // Cardinalities:
        //   |A| = 1000
        //   |B| = 10
        //   |C| = 1
        //
        // Under the placeholder cost model:
        //
        // Plan 1: (A ⋈ B) ⋈ C
        //   AB: card = 1000 * 10 = 10_000
        //       cost = 1000 + 10 + 10_000 = 11_010
        //   ABC: card = 10_000 * 1 = 10_000
        //        cost = 11_010 + 1 + 10_000 = 22_021
        //
        // Plan 2: (B ⋈ C) ⋈ A
        //   BC: card = 10 * 1 = 10
        //       cost = 10 + 1 + 10 = 21
        //   ABC: card = 10 * 1000 = 10_000
        //        cost = 21 + 1000 + 10_000 = 11_021
        //
        // So the optimal plan is (B ⋈ C) ⋈ A with cost 11_021.

        let rel_a = JoinRelation {
            id: 0,
            name: "A".to_string(),
            plan: dummy_scan("A"),
            filters: vec![],
        };
        let rel_b = JoinRelation {
            id: 1,
            name: "B".to_string(),
            plan: dummy_scan("B"),
            filters: vec![],
        };
        let rel_c = JoinRelation {
            id: 2,
            name: "C".to_string(),
            plan: dummy_scan("C"),
            filters: vec![],
        };

        let relations = vec![rel_a, rel_b, rel_c];

        let edges = vec![
            // A <-> B
            JoinEdge {
                left: 0,
                right: 1,
                on: vec![],
            },
            // B <-> C
            JoinEdge {
                left: 1,
                right: 2,
                on: vec![],
            },
        ];

        let predicates = vec![];

        let graph = JoinGraph { relations, edges, predicates };

        let base = [1000.0_f64, 10.0_f64, 1.0_f64];

        // let result =
        //     dphyp_optimize_join_graph(&graph, Some(&base)).expect("expected a DP result");
        let model = SimpleCardinalityCostModel {
            base_cardinalities: Some(&base),
        };

        let result =
            dphyp_optimize_join_graph(&graph, &model).expect("expected a DP result");


        assert_eq!(result.full_subset, 0b111);

        let root = result
            .plans
            .get(&result.full_subset)
            .expect("plan for full subset {A,B,C}");

        assert!((root.cost - 11_021.0).abs() < 1e-6);
        assert!(root.left.is_some() && root.right.is_some());
    }

    /// Compute the cost and cardinality of `plan` using the same
    /// `DPhypCostModel` as the DP. Returns (cost, cardinality, subset_mask).
    ///
    /// `subset_mask` is a bitset over `graph.relations`, so we can
    /// confirm it matches the full_DP_subset.
    fn naive_cost_for_plan<M: DPhypCostModel>(
        plan: &LogicalPlan,
        graph: &JoinGraph,
        model: &M,
    ) -> (f64, f64, BitSet) {
        match plan {
            LogicalPlan::Join(j) => {
                let (cost_l, card_l, mask_l) =
                    naive_cost_for_plan(&j.left, graph, model);
                let (cost_r, card_r, mask_r) =
                    naive_cost_for_plan(&j.right, graph, model);

                // create ephemeral entries so we can call join_cost_cardinality
                let left_entry = DPhypPlanEntry {
                    subset: mask_l,
                    left: None,
                    right: None,
                    cost: cost_l,
                    cardinality: card_l,
                };
                let right_entry = DPhypPlanEntry {
                    subset: mask_r,
                    left: None,
                    right: None,
                    cost: cost_r,
                    cardinality: card_r,
                };

                let (join_cost, result_card) =
                    model.join_cost_cardinality(&left_entry, &right_entry, graph);
                let total_cost = cost_l + cost_r + join_cost;
                (total_cost, result_card, mask_l | mask_r)
            }
            // we expect only Join + TableScan/SubqueryAlias in these tests
            _ => {
                let (tref, _) = relation_ref_and_name(plan)
                    .expect("expected base relation in naive_cost_for_plan");

                // find corresponding relation id in graph
                let rel = graph
                    .relations
                    .iter()
                    .find(|r| {
                        if let Some((rel_tref, _)) = relation_ref_and_name(&r.plan) {
                            rel_tref == tref
                        } else {
                            false
                        }
                    })
                    .expect("relation not found in JoinGraph for naive cost");

                let (base_cost, base_card) =
                    model.base_cost_cardinality(rel.id, graph);
                (base_cost, base_card, 1u64 << rel.id)
            }
        }
    }

    #[test]
    fn extracted_three_way_join_is_not_worse_than_original_tree() {
        // Build logical plan: ((A ⋈ B) ⋈ C)
        let a_scan = dummy_scan("A");
        let b_scan = dummy_scan("B");
        let c_scan = dummy_scan("C");

        // A ⋈ B on A.id = B.id
        let ab_join = LogicalPlanBuilder::from(a_scan)
            .join(
                b_scan,
                JoinType::Inner,
                (
                    vec![Column::new(Some(TableReference::from("A")), "id")],
                    vec![Column::new(Some(TableReference::from("B")), "id")],
                ),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        // (A ⋈ B) ⋈ C on B.id = C.id  (gives edges A-B and B-C)
        let abc_join = LogicalPlanBuilder::from(ab_join)
            .join(
                c_scan,
                JoinType::Inner,
                (
                    vec![Column::new(Some(TableReference::from("B")), "id")],
                    vec![Column::new(Some(TableReference::from("C")), "id")],
                ),
                None,
            )
            .unwrap()
            .build()
            .unwrap();


        // 1) Extract join graph from the plan (Commit 1)
        let graph = extract_join_graph(&abc_join)
            .expect("extract_join_graph should succeed")
            .expect("expected a join graph for three-way join");

        assert_eq!(graph.relations.len(), 3);
        assert!(graph.edges.len() >= 2); // at least 2 join predicates

        // 2) Build base cardinalities per extracted relation id
        //
        // We skew the cardinalities on purpose:
        // |A| = 1000, |B| = 10, |C| = 1
        let mut base = vec![1.0_f64; graph.relations.len()];
        for rel in &graph.relations {
            base[rel.id] = match rel.name.as_str() {
                "A" | "a" => 1000.0,
                "B" | "b" => 10.0,
                "C" | "c" => 1.0,
                other => panic!("unexpected relation name in test: {other}"),
            };
        }

        let model = SimpleCardinalityCostModel {
            base_cardinalities: Some(&base),
        };

        // 3) Run DPhyp-style DP (Commit 2)
        let result = dphyp_optimize_join_graph(&graph, &model)
            .expect("expected a DP result for three-way join");

        // 4) Compute the cost of the *original* join tree using the same model
        let (naive_cost, naive_card, naive_mask) =
            naive_cost_for_plan(&abc_join, &graph, &model);

        let root = result
            .plans
            .get(&result.full_subset)
            .expect("plan for full subset {A,B,C}");

        // The DP plan should cover the same set of relations
        assert_eq!(naive_mask, result.full_subset);

        // And the DP plan should be no more expensive than the original plan
        assert!(
            root.cost <= naive_cost + 1e-6,
            "DP cost {} should be <= naive cost {} (card DP = {}, card naive = {})",
            root.cost,
            naive_cost,
            root.cardinality,
            naive_card
        );
    }

    /// Collect all 2-element subsets (bitmasks) that appear as internal
    /// nodes in the chosen DP plan for `subset`.
    fn collect_pair_subsets(
        result: &DPhypResult,
        subset: BitSet,
        out: &mut Vec<BitSet>,
    ) {
        let entry = match result.plans.get(&subset) {
            Some(e) => e,
            None => return,
        };

        if let (Some(l), Some(r)) = (entry.left, entry.right) {
            if bit_count(l) == 2 {
                out.push(l);
            } else if bit_count(l) > 2 {
                collect_pair_subsets(result, l, out);
            }

            if bit_count(r) == 2 {
                out.push(r);
            } else if bit_count(r) > 2 {
                collect_pair_subsets(result, r, out);
            }
        }
    }

    #[test]
    fn extracted_four_way_chain_uses_smallest_pair_in_optimal_plan() {
        // Build logical plan: (((A ⋈ B) ⋈ C) ⋈ D)
        let a_scan = dummy_scan("A");
        let b_scan = dummy_scan("B");
        let c_scan = dummy_scan("C");
        let d_scan = dummy_scan("D");

        // A ⋈ B on A.id = B.id
        let ab = LogicalPlanBuilder::from(a_scan)
            .join(
                b_scan,
                JoinType::Inner,
                (
                    vec![Column::new(Some(TableReference::from("A")), "id")],
                    vec![Column::new(Some(TableReference::from("B")), "id")],
                ),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        // (A ⋈ B) ⋈ C on B.id = C.id  (chain A-B-C)
        let abc = LogicalPlanBuilder::from(ab)
            .join(
                c_scan,
                JoinType::Inner,
                (
                    vec![Column::new(Some(TableReference::from("B")), "id")],
                    vec![Column::new(Some(TableReference::from("C")), "id")],
                ),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        // ((A ⋈ B) ⋈ C) ⋈ D on C.id = D.id  (chain B-C-D, so edges A-B, B-C, C-D)
        let abcd = LogicalPlanBuilder::from(abc)
            .join(
                d_scan,
                JoinType::Inner,
                (
                    vec![Column::new(Some(TableReference::from("C")), "id")],
                    vec![Column::new(Some(TableReference::from("D")), "id")],
                ),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        // 1) Extract join graph
        let graph = extract_join_graph(&abcd)
            .expect("extract_join_graph should succeed")
            .expect("expected a join graph for four-way join");

        assert_eq!(graph.relations.len(), 4);

        // 2) Assign base cardinalities such that C and D are *clearly* the best pair:
        //
        //   |A| = 1000
        //   |B| = 100
        //   |C| = 1
        //   |D| = 1
        //
        // Under the SimpleCardinalityCostModel, joining C ⋈ D first
        // should be very attractive compared to any pair involving A or B.
        let mut base = vec![1.0_f64; graph.relations.len()];
        for rel in &graph.relations {
            base[rel.id] = match rel.name.as_str() {
                "A" | "a" => 1000.0,
                "B" | "b" => 100.0,
                "C" | "c" => 1.0,
                "D" | "d" => 1.0,
                other => panic!("unexpected relation name in test: {other}"),
            };
        }

        let model = SimpleCardinalityCostModel {
            base_cardinalities: Some(&base),
        };

        // 3) Run DP
        let result = dphyp_optimize_join_graph(&graph, &model)
            .expect("expected a DP result for four-way join");

        // 4) Compute naive cost for sanity
        let (naive_cost, _, naive_mask) =
            naive_cost_for_plan(&abcd, &graph, &model);
        assert_eq!(
            naive_mask, result.full_subset,
            "naive plan and DP plan should span the same subset"
        );

        let root = result
            .plans
            .get(&result.full_subset)
            .expect("plan for full subset {A,B,C,D}");

        // DP should not be worse than naive
        assert!(
            root.cost <= naive_cost + 1e-6,
            "DP cost {} should be <= naive cost {}",
            root.cost,
            naive_cost
        );

        // 5) Verify that the optimal plan *actually* uses {C,D} as a binary join step.
        //
        // Find the ids for C and D in the extracted graph.
        let mut id_c = None;
        let mut id_d = None;
        for rel in &graph.relations {
            match rel.name.as_str() {
                "C" | "c" => id_c = Some(rel.id),
                "D" | "d" => id_d = Some(rel.id),
                _ => {}
            }
        }
        let id_c = id_c.expect("relation C must be present");
        let id_d = id_d.expect("relation D must be present");

        let mask_cd: BitSet = (1u64 << id_c) | (1u64 << id_d);

        let mut pair_subsets = Vec::new();
        collect_pair_subsets(&result, result.full_subset, &mut pair_subsets);

        assert!(
            pair_subsets.contains(&mask_cd),
            "optimal DP plan should include {{C,D}} as a 2-way join; got pairs {:?}",
            pair_subsets
                .into_iter()
                .map(|m| format!("{:#b}", m))
                .collect::<Vec<_>>()
        );
    }

    use super::*;
    use std::collections::HashSet;
    use datafusion_common::{Column, TableReference};

    /// Collect an unordered set of relation names from a JoinGraph
    fn relation_name_set(graph: &JoinGraph) -> HashSet<String> {
        graph
            .relations
            .iter()
            .map(|r| r.name.clone())
            .collect::<HashSet<_>>()
    }

    /// Collect an unordered set of edges as (min(name1, name2), max(name1, name2)).
    fn unordered_edge_name_set(graph: &JoinGraph) -> HashSet<(String, String)> {
        let mut set = HashSet::new();

        for edge in &graph.edges {
            let left_name = &graph.relations[edge.left].name;
            let right_name = &graph.relations[edge.right].name;

            let (a, b) = if left_name <= right_name {
                (left_name.clone(), right_name.clone())
            } else {
                (right_name.clone(), left_name.clone())
            };

            set.insert((a, b));
        }

        set
    }

    /// Helper to build a Column(Expr) for `tref.col_name`.
    fn col_ref(tref: &TableReference, col_name: &str) -> Expr {
        Expr::Column(Column::new(Some(tref.clone()), col_name))
    }

    /// 3-way chain a - b - c:
    ///
    ///   a.id = b.id
    ///   b.id = c.id
    ///
    /// Run:
    ///   JoinGraph -> DP -> LogicalPlan -> extract_join_graph
    ///
    /// and check that relations and edges are preserved (as sets).
    #[test]
    fn dphyp_reconstruction_roundtrips_three_way_graph() {
        // base relations: a, b, c
        let a_plan = dummy_scan("a");
        let b_plan = dummy_scan("b");
        let c_plan = dummy_scan("c");

        let a_ref = TableReference::bare("a");
        let b_ref = TableReference::bare("b");
        let c_ref = TableReference::bare("c");

        let relations = vec![
            JoinRelation {
                id: 0,
                name: "a".to_string(),
                plan: a_plan,
                filters: vec![],
            },
            JoinRelation {
                id: 1,
                name: "b".to_string(),
                plan: b_plan,
                filters: vec![],
            },
            JoinRelation {
                id: 2,
                name: "c".to_string(),
                plan: c_plan,
                filters: vec![],
            },
        ];

        // edges: a - b, b - c
        let edges = vec![
            JoinEdge {
                left: 0,
                right: 1,
                on: vec![(col_ref(&a_ref, "id"), col_ref(&b_ref, "id"))],
            },
            JoinEdge {
                left: 1,
                right: 2,
                on: vec![(col_ref(&b_ref, "id"), col_ref(&c_ref, "id"))],
            },
        ];

        let predicates = vec![];

        let graph = JoinGraph { relations, edges, predicates };

        // Run DP with the simple placeholder cost model
        let model = SimpleCardinalityCostModel { base_cardinalities: None };
        let dp = dphyp_optimize_join_graph(&graph, &model)
            .expect("DP optimization should succeed for connected 3-way graph");

        // Rebuild a logical plan from the DP result
        let rebuilt_plan =
            build_best_join_plan_from_dphyp(&graph, &dp).expect("rebuild plan");

        // Extract a JoinGraph from the rebuilt plan using Commit 1
        let rebuilt_graph = extract_join_graph(&rebuilt_plan)
            .expect("extract_join_graph on rebuilt plan should succeed")
            .expect("rebuilt plan should yield some JoinGraph");

        // Compare relation sets
        let orig_relations = relation_name_set(&graph);
        let rebuilt_relations = relation_name_set(&rebuilt_graph);
        assert_eq!(orig_relations, rebuilt_relations);

        // Compare unordered edge sets by relation name
        let orig_edges = unordered_edge_name_set(&graph);
        let rebuilt_edges = unordered_edge_name_set(&rebuilt_graph);
        assert_eq!(orig_edges, rebuilt_edges);
    }

    /// 4-way chain a - b - c - d:
    ///
    ///   a.id = b.id
    ///   b.id = c.id
    ///   c.id = d.id
    ///
    /// Same round-trip test as above but with a larger island.
    #[test]
    fn dphyp_reconstruction_roundtrips_four_way_graph() {
        let a_plan = dummy_scan("a");
        let b_plan = dummy_scan("b");
        let c_plan = dummy_scan("c");
        let d_plan = dummy_scan("d");

        let a_ref = TableReference::bare("a");
        let b_ref = TableReference::bare("b");
        let c_ref = TableReference::bare("c");
        let d_ref = TableReference::bare("d");

        let relations = vec![
            JoinRelation {
                id: 0,
                name: "a".to_string(),
                plan: a_plan,
                filters: vec![],
            },
            JoinRelation {
                id: 1,
                name: "b".to_string(),
                plan: b_plan,
                filters: vec![],
            },
            JoinRelation {
                id: 2,
                name: "c".to_string(),
                plan: c_plan,
                filters: vec![],
            },
            JoinRelation {
                id: 3,
                name: "d".to_string(),
                plan: d_plan,
                filters: vec![],
            },
        ];

        let edges = vec![
            JoinEdge {
                left: 0,
                right: 1,
                on: vec![(col_ref(&a_ref, "id"), col_ref(&b_ref, "id"))],
            },
            JoinEdge {
                left: 1,
                right: 2,
                on: vec![(col_ref(&b_ref, "id"), col_ref(&c_ref, "id"))],
            },
            JoinEdge {
                left: 2,
                right: 3,
                on: vec![(col_ref(&c_ref, "id"), col_ref(&d_ref, "id"))],
            },
        ];

        let predicates = vec![];

        let graph = JoinGraph { relations, edges, predicates };

        let model = SimpleCardinalityCostModel { base_cardinalities: None };
        let dp = dphyp_optimize_join_graph(&graph, &model)
            .expect("DP optimization should succeed for connected 4-way graph");

        let rebuilt_plan =
            build_best_join_plan_from_dphyp(&graph, &dp).expect("rebuild plan");

        let rebuilt_graph = extract_join_graph(&rebuilt_plan)
            .expect("extract_join_graph on rebuilt plan should succeed")
            .expect("rebuilt plan should yield some JoinGraph");

        let orig_relations = relation_name_set(&graph);
        let rebuilt_relations = relation_name_set(&rebuilt_graph);
        assert_eq!(orig_relations, rebuilt_relations);

        let orig_edges = unordered_edge_name_set(&graph);
        let rebuilt_edges = unordered_edge_name_set(&rebuilt_graph);
        assert_eq!(orig_edges, rebuilt_edges);
    }

    fn column_with_rel(rel: &str, name: &str) -> Expr {
        Expr::Column(Column {
            relation: Some(TableReference::bare(rel)),
            name: name.to_string(),
            spans: Spans::new()
        })
    }

    #[test]
    fn rel_mask_for_multi_relation_expr() {
        // Build a dummy GraphBuilder with 4 base relations a, b, c, d
        let mut gb = GraphBuilder::default();
        for rel in ["a", "b", "c", "d"] {
            gb.add_base_relation(&dummy_scan(rel));
        }

        let expr_left = column_with_rel("a", "id") + column_with_rel("b", "id");
        let expr_right = column_with_rel("c", "id") + column_with_rel("d", "id");

        let (left_mask, left_single) =
            gb.rel_mask_and_single_id_for_expr(&expr_left).unwrap();
        let (right_mask, right_single) =
            gb.rel_mask_and_single_id_for_expr(&expr_right).unwrap();

        // left side touches a, b; right side touches c, d
        assert!(left_single.is_none());
        assert!(right_single.is_none());

        // Check masks have the expected number of bits set.
        assert_eq!(bit_count(left_mask), 2);
        assert_eq!(bit_count(right_mask), 2);

        let full = left_mask | right_mask;
        assert_eq!(bit_count(full), 4);
    }

    #[test]
    fn extract_three_way_hyperedge_predicate_from_plan() {
        use datafusion_common::NullEquality;
        use datafusion_expr::JoinConstraint;
        use datafusion_expr::logical_plan::Join;

        // Base scans
        let a_scan = dummy_scan("a");
        let b_scan = dummy_scan("b");
        let c_scan = dummy_scan("c");

        // Left child: a ⋈ b on a.id = b.id  (simple equi join)
        let ab_join = LogicalPlanBuilder::from(a_scan.clone())
            .join(
                b_scan.clone(),
                JoinType::Inner,
                (vec!["id"], vec!["id"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        let left = Arc::new(ab_join);
        let right = Arc::new(c_scan.clone());

        // Hyperedge predicate: a.id + b.id = c.id
        let expr_left = column_with_rel("a", "id") + column_with_rel("b", "id");
        let expr_right = column_with_rel("c", "id");

        let join = Join::try_new(
            left,
            right,
            vec![(expr_left.clone(), expr_right.clone())],
            None,
            JoinType::Inner,
            JoinConstraint::On,
            NullEquality::NullEqualsNull,
        )
            .unwrap();

        let plan = LogicalPlan::Join(join);

        let graph = extract_join_graph(&plan)
            .expect("extract_join_graph should succeed")
            .expect("plan should yield Some(JoinGraph)");

        // We have three base relations
        assert_eq!(graph.relations.len(), 3);

        // From the inner a=b equi-join we get one binary edge.
        assert_eq!(graph.edges.len(), 1);

        // Predicates: one for a.id = b.id, one for a.id + b.id = c.id
        assert_eq!(graph.predicates.len(), 2);

        let mask_counts: Vec<_> = graph
            .predicates
            .iter()
            .map(|p| bit_count(p.rel_mask))
            .collect();

        // One predicate touches exactly {a,b}
        assert!(
            mask_counts.contains(&2),
            "expected binary predicate for a.id = b.id"
        );
        // One predicate touches {a,b,c}
        assert!(
            mask_counts.contains(&3),
            "expected hyperedge predicate touching a,b,c"
        );
    }

    #[test]
    fn dphyp_three_way_hyperedge_graph_produces_plan() {
        // Base relations A, B, C
        let rel_a = JoinRelation {
            id: 0,
            name: "A".to_string(),
            plan: dummy_scan("A"),
            filters: vec![],
        };
        let rel_b = JoinRelation {
            id: 1,
            name: "B".to_string(),
            plan: dummy_scan("B"),
            filters: vec![],
        };
        let rel_c = JoinRelation {
            id: 2,
            name: "C".to_string(),
            plan: dummy_scan("C"),
            filters: vec![],
        };

        let relations = vec![rel_a, rel_b, rel_c];

        // Build a hyperedge for A.id + B.id = C.id
        let a_ref = TableReference::bare("A");
        let b_ref = TableReference::bare("B");
        let c_ref = TableReference::bare("C");

        let expr_left = col_ref(&a_ref, "id") + col_ref(&b_ref, "id");
        let expr_right = col_ref(&c_ref, "id");

        let hyper_mask: BitSet = (1u64 << 0) | (1u64 << 1) | (1u64 << 2);

        let predicates = vec![JoinPredicate {
            rel_mask: hyper_mask,
            expr: expr_left.eq(expr_right),
        }];

        // No simple edges at all: connectivity comes purely from the hyperedge.
        let edges = vec![];

        let graph = JoinGraph {
            relations,
            edges,
            predicates,
        };

        // Some arbitrary base cardinalities, just to exercise the model.
        let base = [100.0_f64, 10.0_f64, 1.0_f64];
        let model = SimpleCardinalityCostModel {
            base_cardinalities: Some(&base),
        };

        let result = dphyp_optimize_join_graph(&graph, &model)
            .expect("DP optimization should succeed for connected hypergraph");

        assert_eq!(result.full_subset, hyper_mask);

        // We should have entries for each singleton and for the full set.
        for mask in [1u64, 2u64, 4u64, hyper_mask] {
            assert!(
                result.plans.contains_key(&mask),
                "expected plan entry for subset mask {:#b}",
                mask
            );
        }
    }

    #[test]
    fn predicate_selectivity_influences_three_way_chain_order() {
        // Graph: A -- B -- C with two predicates:
        //   p0: A.id = B.id  (very selective)
        //   p1: B.id = C.id  (less selective)
        //
        // Base cardinalities all equal (so only selectivity matters):
        //   |A| = |B| = |C| = 1000
        //
        // With the PredicateSelectivityCostModel, joining A⋈B first
        // (using p0 sel = 1e-4) should be cheaper than joining B⋈C
        // first (p1 sel = 1e-2).

        // Base relations A, B, C
        let rel_a = JoinRelation {
            id: 0,
            name: "A".to_string(),
            plan: dummy_scan("A"),
            filters: vec![],
        };
        let rel_b = JoinRelation {
            id: 1,
            name: "B".to_string(),
            plan: dummy_scan("B"),
            filters: vec![],
        };
        let rel_c = JoinRelation {
            id: 2,
            name: "C".to_string(),
            plan: dummy_scan("C"),
            filters: vec![],
        };

        let relations = vec![rel_a, rel_b, rel_c];

        let a_ref = TableReference::bare("A");
        let b_ref = TableReference::bare("B");
        let c_ref = TableReference::bare("C");

        // Binary edges: A-B and B-C
        let edges = vec![
            JoinEdge {
                left: 0,
                right: 1,
                on: vec![(col_ref(&a_ref, "id"), col_ref(&b_ref, "id"))],
            },
            JoinEdge {
                left: 1,
                right: 2,
                on: vec![(col_ref(&b_ref, "id"), col_ref(&c_ref, "id"))],
            },
        ];

        // Hypergraph predicates matching the same equalities
        let predicates = vec![
            // p0: A.id = B.id  (mask {A,B})
            JoinPredicate {
                rel_mask: (1u64 << 0) | (1u64 << 1),
                expr: col_ref(&a_ref, "id").eq(col_ref(&b_ref, "id")),
            },
            // p1: B.id = C.id  (mask {B,C})
            JoinPredicate {
                rel_mask: (1u64 << 1) | (1u64 << 2),
                expr: col_ref(&b_ref, "id").eq(col_ref(&c_ref, "id")),
            },
        ];

        let graph = JoinGraph {
            relations,
            edges,
            predicates,
        };

        // Base cardinalities: all equal
        let base = [1000.0_f64, 1000.0_f64, 1000.0_f64];

        // Predicate selectivities:
        //   sel(p0: A=B) = 1e-4  (very selective)
        //   sel(p1: B=C) = 1e-2  (less selective)
        let predicate_sels = [1e-4_f64, 1e-2_f64];

        let model = PredicateSelectivityCostModel {
            base_cardinalities: Some(&base),
            predicate_selectivities: Some(&predicate_sels),
        };

        let result =
            dphyp_optimize_join_graph(&graph, &model).expect("expected DP result");

        assert_eq!(result.full_subset, 0b111);

        // Collect all 2-element subsets used as internal join nodes.
        let mut pair_subsets = Vec::new();
        collect_pair_subsets(&result, result.full_subset, &mut pair_subsets);

        // For a 3-way join there should be exactly one 2-relation internal subset.
        assert_eq!(pair_subsets.len(), 1);

        // Expected best pair is {A,B} (ids 0 and 1)
        let mask_ab: BitSet = (1u64 << 0) | (1u64 << 1);
        assert_eq!(
            pair_subsets[0], mask_ab,
            "expected {{A,B}} to be the first join pair due to more selective predicate; \
             got pair mask {:#b}",
            pair_subsets[0]
        );
    }

    use datafusion_expr::Operator;
    #[test]
    fn filter_only_two_way_join_creates_hyperedge_and_is_reconstructible() {
        // Build two base relations A and B
        let a_scan = dummy_scan("a");
        let b_scan = dummy_scan("b");

        // Build a non-equi join predicate that touches both A and B.
        // We model "A.cola contains B.colb" as a LIKE for simplicity:
        //
        //   a.cola LIKE b.colb
        //
        // The hypergraph machinery only cares that the expression
        // references columns from *both* relations.
        let a_col_expr = Expr::Column(Column {
            relation: Some(TableReference::bare("a")),
            name: "cola".to_string(),
            spans: Spans::new()
        });
        let b_col_expr = Expr::Column(Column {
            relation: Some(TableReference::bare("b")),
            name: "colb".to_string(),
            spans: Spans::new()
        });

        let filter_expr = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(a_col_expr),
            op: Operator::LikeMatch,
            right: Box::new(b_col_expr),
        });

        // Build an INNER join with:
        //   - empty ON key lists
        //   - the non-equi predicate in `filter`
        let join_plan = LogicalPlanBuilder::from(a_scan)
            .join(
                b_scan,
                JoinType::Inner,
                (Vec::<Column>::new(), Vec::<Column>::new()),
                Some(filter_expr.clone()),
            )
            .unwrap()
            .build()
            .unwrap();

        // ---- extract join graph ----
        let graph = extract_join_graph(&join_plan)
            .expect("extract_join_graph should not error")
            .expect("expected a join graph for simple inner join");

        // We should have exactly two base relations, no equi-edges,
        // and one predicate hyperedge over both relations.
        assert_eq!(graph.relations.len(), 2);
        assert!(graph.edges.is_empty(), "no equi-join edges expected");
        assert_eq!(graph.predicates.len(), 1);

        let pred = &graph.predicates[0];
        // Predicate must reference exactly the two relations in this island.
        assert_eq!(pred.rel_mask.count_ones(), 2);

        // ---- run DPhyp and reconstruct plan ----
        let cost_model = SimpleCardinalityCostModel { base_cardinalities: None };
        let dp = dphyp_optimize_join_graph(&graph, &cost_model)
            .expect("DP optimization should succeed for connected 2-way graph");

        // Rebuild the best plan from the DP table.
        let rebuilt = build_best_join_plan_from_dphyp(&graph, &dp)
            .expect("reconstruction should succeed");

        // The rebuilt plan should be an INNER join with:
        //   - no ON keys
        //   - a non-empty filter (our predicate)
        match rebuilt {
            LogicalPlan::Join(Join {
                                  join_type,
                                  join_constraint,
                                  on,
                                  filter,
                                  ..
                              }) => {
                assert_eq!(join_type, JoinType::Inner);
                assert_eq!(join_constraint, JoinConstraint::On);
                assert!(
                    on.is_empty(),
                    "non-equi join should not have equi key pairs in `on`"
                );
                assert!(
                    filter.is_some(),
                    "filter-only join must carry the predicate in `filter`"
                );
            }
            other => panic!("expected Join plan after reconstruction, got: {other:?}"),
        }
    }

    use std::collections::HashMap;

    struct NamedCardinalityCostModel {
        /// Cardinalities keyed by relation name, e.g. "a", "b", "c".
        cards: HashMap<String, f64>,
    }

    impl DPhypCostModel for NamedCardinalityCostModel {
        fn base_cost_cardinality(
            &self,
            rel_id: usize,
            graph: &JoinGraph,
        ) -> (f64, f64) {
            let name = &graph.relations[rel_id].name;
            let card = self.cards.get(name).copied().unwrap_or(1.0);
            (card, card)
        }

        fn join_cost_cardinality(
            &self,
            left: &DPhypPlanEntry,
            right: &DPhypPlanEntry,
            _graph: &JoinGraph,
        ) -> (f64, f64) {
            let result_card = left.cardinality * right.cardinality;
            let join_cost = result_card;
            (join_cost, result_card)
        }
    }


    #[test]
    fn optimize_joins_recursively_rewrites_three_way_chain_under_filter() {
        use datafusion_expr::{lit, logical_plan::Filter};

        let a_scan = dummy_scan("a");
        let b_scan = dummy_scan("b");
        let c_scan = dummy_scan("c");

        // First join: a.id = b.id
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

        // Second join: (a ⋈ b) ⋈ c, explicitly b.id = c.id
        let left_deep = LogicalPlanBuilder::from(ab_join)
            .join(
                c_scan,
                JoinType::Inner,
                (vec!["b.id"], vec!["id"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        // Wrap the join island in a Filter
        let filtered = LogicalPlan::Filter(
            Filter::try_new(lit(true), Arc::new(left_deep.clone())).unwrap(),
        );

        // Cardinalities keyed by relation name:
        //   |a| = 1000, |b| = 10, |c| = 1
        let mut cards = HashMap::new();
        cards.insert("a".to_string(), 1000.0);
        cards.insert("b".to_string(), 10.0);
        cards.insert("c".to_string(), 1.0);
        let model = NamedCardinalityCostModel { cards };

        // Sanity: DPhyp on the original left-deep chain should NOT leave it as-is.
        let graph_orig = extract_join_graph(&left_deep)
            .unwrap()
            .expect("left-deep chain should be recognized as a join island");
        let dp_orig =
            dphyp_optimize_join_graph(&graph_orig, &model).expect("DP should succeed");
        let best_orig =
            build_best_join_plan_from_dphyp(&graph_orig, &dp_orig).unwrap();
        assert_ne!(
            best_orig, left_deep,
            "left-deep chain should not be cheapest with these cardinalities"
        );

        // Now run the recursive optimizer on the *filtered* plan.
        let optimized = optimize_joins_in_plan(&filtered, &model).unwrap();

        // Still a Filter at the root.
        let join_under_filter = match optimized {
            LogicalPlan::Filter(f) => (*f.input).clone(),
            other => panic!("expected Filter at root, got {other:?}"),
        };

        // join_under_filter should be a Join whose *left* child is a Join
        // between exactly {b,c} (in any order).
        let (left_child, right_child) = match &join_under_filter {
            LogicalPlan::Join(j) => (j.left.as_ref(), j.right.as_ref()),
            other => panic!("expected Join under Filter, got {other:?}"),
        };

        // Collect leaf relation names under the left child of the top join.
        fn leaf_relation_names(plan: &LogicalPlan) -> Vec<String> {
            let mut names = Vec::new();
            plan.apply(|node| {
                if let LogicalPlan::TableScan(ts) = node {
                    names.push(ts.table_name.to_string());
                }
                Ok(TreeNodeRecursion::Continue)
            })
                .unwrap();
            names
        }

        let left_leaf_names = leaf_relation_names(left_child);

        // We expect the DP to join b and c first (the two smallest).
        assert!(
            left_leaf_names.contains(&"b".to_string())
                && left_leaf_names.contains(&"c".to_string())
                && left_leaf_names.len() == 2,
            "expected left child of top join to be the join of {{b,c}}, got leaves {:?}",
            left_leaf_names
        );

        // And the right child should be the remaining relation a.
        let right_leaf_names = leaf_relation_names(right_child);
        assert_eq!(
            right_leaf_names,
            vec!["a".to_string()],
            "expected right child of top join to be {{a}}, got leaves {:?}",
            right_leaf_names
        );
    }

    #[test]
    fn optimize_joins_recursively_rewrites_three_way_chain_under_aggregate() {
        use std::collections::HashMap;
        use datafusion_expr::col;

        // Base scans with names "a", "b", "c" so they match JoinRelation.name
        let a_scan = dummy_scan("a");
        let b_scan = dummy_scan("b");
        let c_scan = dummy_scan("c");

        // First join: a.id = b.id (unambiguous)
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

        // Second join: (a ⋈ b) ⋈ c, explicitly b.id = c.id to avoid ambiguity
        let left_deep = LogicalPlanBuilder::from(ab_join)
            .join(
                c_scan,
                JoinType::Inner,
                (vec!["b.id"], vec!["id"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        // Wrap the 3-way join in an Aggregate to simulate a non-join operator
        // above the island. Grouping key choice is arbitrary here.
        let aggregated = LogicalPlanBuilder::from(left_deep.clone())
            .aggregate(vec![col("a.id")], Vec::<Expr>::new())
            .unwrap()
            .build()
            .unwrap();

        // Cardinalities chosen so that the optimal plan joins b and c first:
        //   |a| = 1000, |b| = 10, |c| = 1
        let mut cards = HashMap::new();
        cards.insert("a".to_string(), 1000.0);
        cards.insert("b".to_string(), 10.0);
        cards.insert("c".to_string(), 1.0);
        let model = NamedCardinalityCostModel { cards };

        // Sanity: DPhyp over the raw left-deep chain should NOT keep that shape
        let graph_orig = extract_join_graph(&left_deep)
            .unwrap()
            .expect("left-deep chain should be recognized as a join island");
        let dp_orig =
            dphyp_optimize_join_graph(&graph_orig, &model).expect("DP should succeed");
        let best_orig =
            build_best_join_plan_from_dphyp(&graph_orig, &dp_orig).unwrap();

        assert_ne!(
            best_orig, left_deep,
            "left-deep chain should not be cheapest for these cardinalities"
        );

        // Run the recursive optimizer on the plan rooted at Aggregate
        let optimized = optimize_joins_in_plan(&aggregated, &model).unwrap();

        // Shape: still an Aggregate at the root
        let join_under_agg = match optimized {
            LogicalPlan::Aggregate(agg) => (*agg.input).clone(),
            other => panic!("expected Aggregate at root, got {other:?}"),
        };

        // Helper: collect leaf table names under a (possibly nested) join tree
        fn leaf_relation_names(plan: &LogicalPlan, out: &mut Vec<String>) {
            match plan {
                LogicalPlan::Join(j) => {
                    leaf_relation_names(&j.left, out);
                    leaf_relation_names(&j.right, out);
                }
                LogicalPlan::TableScan(scan) => {
                    out.push(scan.table_name.table().to_string());
                }
                _ => {}
            }
        }

        // We only care about which relations are grouped on each side of the
        // top join, not the exact filter duplication structure.
        let root_join = match &join_under_agg {
            LogicalPlan::Join(j) => j,
            other => panic!("expected Join under Aggregate, got {other:?}"),
        };

        let mut left_names = Vec::new();
        let mut right_names = Vec::new();
        leaf_relation_names(&root_join.left, &mut left_names);
        leaf_relation_names(&root_join.right, &mut right_names);
        left_names.sort();
        right_names.sort();

        // With |a| >> |b|,|c| and the Simple/NamedCardinality cost model,
        // DPhyp should pick (b ⋈ c) first, then join with a.
        assert!(
            (left_names == vec!["a".to_string()] && right_names == vec!["b".to_string(), "c".to_string()])
                || (right_names == vec!["a".to_string()] && left_names == vec!["b".to_string(), "c".to_string()]),
            "expected one side of top join to contain {{b,c}} and the other {{a}}, \
             got left={left_names:?}, right={right_names:?}"
        );
    }

    #[test]
    fn optimize_joins_recursively_respects_outer_join_boundary() {
        // Inner-join island on a, b, c
        let a_scan = dummy_scan("a");
        let b_scan = dummy_scan("b");
        let c_scan = dummy_scan("c");
        let d_scan = dummy_scan("d");

        // (a ⋈ b) on a.id = b.id
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

        // ((a ⋈ b) ⋈ c) with b.id = c.id
        let inner_chain = LogicalPlanBuilder::from(ab_join)
            .join(
                c_scan,
                JoinType::Inner,
                (vec!["b.id"], vec!["id"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        // Wrap the inner island on the left side of a LEFT join with d:
        //   ( (a ⋈ b ⋈ c) LEFT JOIN d ON c.id = d.id )
        let full_plan = LogicalPlanBuilder::from(inner_chain.clone())
            .join(
                d_scan,
                JoinType::Left,
                (vec!["c.id"], vec!["id"]),
                None,
            )
            .unwrap()
            .build()
            .unwrap();

        // Cardinalities so that the inner island prefers (b ⋈ c) first:
        //   |a| = 1000, |b| = 10, |c| = 1, |d| arbitrary
        let mut cards = HashMap::new();
        cards.insert("a".to_string(), 1000.0);
        cards.insert("b".to_string(), 10.0);
        cards.insert("c".to_string(), 1.0);
        cards.insert("d".to_string(), 5.0);
        let model = NamedCardinalityCostModel { cards };

        // Run the recursive optimizer on the full plan
        let optimized = optimize_joins_in_plan(&full_plan, &model).unwrap();

        // Root must remain a LEFT join (outer-join boundary)
        let (left_child, right_child, join_type) = match &optimized {
            LogicalPlan::Join(j) => (&*j.left, &*j.right, j.join_type),
            other => panic!("expected top-level Join, got {other:?}"),
        };

        assert_eq!(
            join_type,
            JoinType::Left,
            "top-level join should remain a LEFT join boundary"
        );

        // Helper: collect leaf table names under a (possibly nested) subtree
        fn leaf_relation_names(plan: &LogicalPlan) -> Vec<String> {
            let mut names = Vec::new();
            plan.apply(|node| {
                if let LogicalPlan::TableScan(ts) = node {
                    names.push(ts.table_name.table().to_string());
                }
                Ok(TreeNodeRecursion::Continue)
            })
                .unwrap();
            names
        }

        // Right side should still be only table d
        let mut right_names = leaf_relation_names(right_child);
        right_names.sort();
        assert_eq!(
            right_names,
            vec!["d".to_string()],
            "expected right child of LEFT join to be {{d}}, got {right_names:?}"
        );

        // Left side is the inner island over {a,b,c}
        let mut left_names = leaf_relation_names(left_child);
        left_names.sort();
        assert_eq!(
            left_names,
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
            "expected left child of LEFT join to contain {{a,b,c}}, got {left_names:?}"
        );

        // We also expect that somewhere inside that left subtree, there is a
        // sub-join operating exactly on {b,c}, because they are the two
        // smallest relations in our cost model.
        let left_join = match left_child {
            LogicalPlan::Join(j) => j,
            other => panic!("expected inner island to be a Join, got {other:?}"),
        };

        let mut child1_names = leaf_relation_names(&left_join.left);
        let mut child2_names = leaf_relation_names(&left_join.right);
        child1_names.sort();
        child2_names.sort();

        let have_bc_subjoin =
            child1_names == vec!["b".to_string(), "c".to_string()] ||
                child2_names == vec!["b".to_string(), "c".to_string()];

        assert!(
            have_bc_subjoin,
            "expected a sub-join on {{b,c}} inside inner island, \
             got left child leaves={child1_names:?}, right child leaves={child2_names:?}"
        );
    }


}
