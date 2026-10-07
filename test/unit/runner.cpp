// SPDX-License-Identifier: BSD-3-Clause
#include <iostream>
#include <string>
namespace tilemega::tests::target_spec_test { int TestTargetSpec(int, char**); }
namespace tilemega::tests::coupling_types_test { int TestCouplingTypes(int, char**); }
namespace tilemega::tests::table27_test { int TestTable27(int, char**); }
namespace tilemega::tests::semantics_test { int TestSemantics(int, char**); }
namespace tilemega::tests::dram_floor_test { int TestDramFloor(int, char**); }
namespace tilemega::tests::regime_a_price_test { int TestRegimeAPrice(int, char**); }
namespace tilemega::tests::stage_flow_test { int TestStageFlow(int, char**); }
namespace tilemega::tests::flow_runtime_release_test { int TestFlowRuntimeRelease(int, char**); }
namespace tilemega::tests::incidence_test { int TestIncidence(int, char**); }
namespace tilemega::tests::containment_test { int TestContainment(int, char**); }
namespace tilemega::tests::event_synthesis_test { int TestEventSynthesis(int, char**); }
namespace tilemega::tests::isl_relation_test { int TestIslRelation(int, char**); }
namespace tilemega::tests::runtime_projection_test { int TestRuntimeProjection(int, char**); }
namespace tilemega::tests::relation_bounds_test { int TestRelationBounds(int, char**); }
namespace tilemega::tests::task_element_work_test { int TestTaskElementWork(int, char**); }
namespace tilemega::tests::graph_pattern_test { int TestGraphPattern(int, char**); }
namespace tilemega::tests::model_plan_order_test { int TestModelPlanOrder(int, char**); }
namespace tilemega::tests::layout_bridge_test { int TestLayoutBridge(int, char**); }
namespace tilemega::tests::cg_attr_test { int TestCgAttr(int, char**); }
namespace tilemega::tests::backend_query_test { int TestBackendQuery(int, char**); }
namespace tilemega::tests::chain_dp_test { int TestChainDp(int, char**); }
namespace tilemega::tests::cache_service_curve_test { int TestCacheServiceCurve(int, char**); }
namespace tilemega::tests::handoff_runtime_projection_test { int TestHandoffRuntimeProjection(int, char**); }
namespace tilemega::tests::handoff_ir_test { int TestHandoffIr(int, char**); }
namespace tilemega::tests::handoff_access_test { int TestHandoffAccess(int, char**); }
namespace tilemega::tests::fusion_access_test { int TestFusionAccess(int, char**); }
namespace tilemega::tests::lane_intersections_test { int TestLaneIntersections(int, char**); }
namespace tilemega::tests::fusion_rewrite_test { int TestFusionRewrite(int, char**); }
namespace tilemega::tests::stage_kappa_test { int TestStageKappa(int, char**); }
namespace tilemega::tests::pipeline_sigma_test { int TestPipelineSigma(int, char**); }
namespace tilemega::tests::fusion_written_price_test { int TestFusionWrittenPrice(int, char**); }
namespace tilemega::tests::fused_runtime_projection_test { int TestFusedRuntimeProjection(int, char**); }
namespace tilemega::tests::exact_runtime_task_graph_test { int TestExactRuntimeTaskGraph(int, char**); }
namespace tilemega::tests::fusion_batch_rewrite_test { int TestFusionBatchRewrite(int, char**); }
namespace tilemega::tests::fiber_sum_test { int TestFiberSum(int, char**); }
namespace tilemega::tests::list_scheduler_test { int TestListScheduler(int, char**); }
namespace tilemega::tests::plan_contract_test { int TestPlanContract(int, char**); }
namespace tilemega::tests::execution_simulator_test { int TestExecutionSimulator(int, char**); }
namespace tilemega::tests::eft_placement_test { int TestEftPlacement(int, char**); }
namespace tilemega::tests::chain_placement_test { int TestChainPlacement(int, char**); }
namespace tilemega::tests::plan_table_test { int TestPlanTable(int, char**); }
namespace tilemega::tests::cluster_labeling_test { int TestClusterLabeling(int, char**); }
namespace tilemega::tests::alignment_propagation_test { int TestAlignmentPropagation(int, char**); }
namespace tilemega::tests::impl_contract_test { int TestImplContract(int, char**); }
namespace tilemega::tests::serving_pruning_test { int TestServingPruning(int, char**); }
namespace tilemega::tests::serving_lag_test { int TestServingLag(int, char**); }
namespace tilemega::tests::serving_task_index_test { int TestServingTaskIndex(int, char**); }
namespace tilemega::tests::serving_model_plan_test { int TestServingModelPlan(int, char**); }
namespace tilemega::tests::serving_token_sets_test { int TestServingTokenSets(int, char**); }
namespace tilemega::tests::serving_import_test { int TestServingImport(int, char**); }
namespace tilemega::tests::forward_frontend_test { int TestForwardFrontend(int, char**); }
namespace tilemega::tests::frontend_import_test { int TestFrontendImport(int, char**); }
namespace tilemega::tests::dm_descriptor_test { int TestDmDescriptor(int, char**); }
namespace tilemega::tests::skeleton_placement_test { int TestSkeletonPlacement(int, char**); }
namespace tilemega::tests::isolated_evaluation_test { int TestIsolatedEvaluation(int, char**); }
namespace tilemega::tests::skeleton_search_isolation_test { int TestSkeletonSearchIsolation(int, char**); }
namespace tilemega::tests::plan_skeleton_test { int TestPlanSkeleton(int, char**); }
namespace tilemega::tests::variant_resource_test { int TestVariantResource(int, char**); }
namespace tilemega::tests::symbolic_oracle_test { int TestSymbolicOracle(int, char**); }
namespace tilemega::tests::operator_classes_test { int TestOperatorClasses(int, char**); }
namespace tilemega::tests::coupling_cache_test { int TestCouplingCache(int, char**); }
namespace tilemega::tests::semantic_lifting_test { int TestSemanticLifting(int, char**); }
namespace tilemega::tests::embedding_plan_test { int TestEmbeddingPlan(int, char**); }
namespace tilemega::tests::wiring_coupling_test { int TestWiringCoupling(int, char**); }
namespace tilemega::tests::degradation_test { int TestDegradation(int, char**); }
int main(int argc, char** argv) {
  struct Entry { char const* name; int (*run)(int, char**); };
  Entry const entries[] = {
    {"target_spec", tilemega::tests::target_spec_test::TestTargetSpec},
    {"coupling_types", tilemega::tests::coupling_types_test::TestCouplingTypes},
    {"table27", tilemega::tests::table27_test::TestTable27},
    {"stage_flow", tilemega::tests::stage_flow_test::TestStageFlow},
    {"flow_runtime_release", tilemega::tests::flow_runtime_release_test::TestFlowRuntimeRelease},
    {"regime_a_price", tilemega::tests::regime_a_price_test::TestRegimeAPrice},
    {"dram_floor", tilemega::tests::dram_floor_test::TestDramFloor},
    {"semantics", tilemega::tests::semantics_test::TestSemantics},
    {"incidence_identity", tilemega::tests::incidence_test::TestIncidence},
    {"containment", tilemega::tests::containment_test::TestContainment},
    {"event_synthesis", tilemega::tests::event_synthesis_test::TestEventSynthesis},
    {"runtime_projection", tilemega::tests::runtime_projection_test::TestRuntimeProjection},
    {"relation_bounds", tilemega::tests::relation_bounds_test::TestRelationBounds},
    {"graph_pattern", tilemega::tests::graph_pattern_test::TestGraphPattern},
    {"task_element_work", tilemega::tests::task_element_work_test::TestTaskElementWork},
    {"isl_relation", tilemega::tests::isl_relation_test::TestIslRelation},
    {"layout_bridge", tilemega::tests::layout_bridge_test::TestLayoutBridge},
    {"cg_attr_roundtrip", tilemega::tests::cg_attr_test::TestCgAttr},
    {"backend_query", tilemega::tests::backend_query_test::TestBackendQuery},
    {"cache_service_curve", tilemega::tests::cache_service_curve_test::TestCacheServiceCurve},
    {"handoff_runtime_projection", tilemega::tests::handoff_runtime_projection_test::TestHandoffRuntimeProjection},
    {"handoff_ir", tilemega::tests::handoff_ir_test::TestHandoffIr},
    {"handoff_access", tilemega::tests::handoff_access_test::TestHandoffAccess},
    {"fusion_access", tilemega::tests::fusion_access_test::TestFusionAccess},
    {"lane_intersections", tilemega::tests::lane_intersections_test::TestLaneIntersections},
    {"fusion_rewrite", tilemega::tests::fusion_rewrite_test::TestFusionRewrite},
    {"stage_kappa", tilemega::tests::stage_kappa_test::TestStageKappa},
    {"pipeline_sigma", tilemega::tests::pipeline_sigma_test::TestPipelineSigma},
    {"fusion_written_price", tilemega::tests::fusion_written_price_test::TestFusionWrittenPrice},
    {"fused_runtime_projection", tilemega::tests::fused_runtime_projection_test::TestFusedRuntimeProjection},
    {"exact_runtime_task_graph", tilemega::tests::exact_runtime_task_graph_test::TestExactRuntimeTaskGraph},
    {"fusion_batch_rewrite", tilemega::tests::fusion_batch_rewrite_test::TestFusionBatchRewrite},
    {"fiber_sum", tilemega::tests::fiber_sum_test::TestFiberSum},
    {"chain_dp", tilemega::tests::chain_dp_test::TestChainDp},
    {"list_scheduler", tilemega::tests::list_scheduler_test::TestListScheduler},
    {"plan_contract", tilemega::tests::plan_contract_test::TestPlanContract},
    {"execution_simulator", tilemega::tests::execution_simulator_test::TestExecutionSimulator},
    {"eft_placement", tilemega::tests::eft_placement_test::TestEftPlacement},
    {"chain_placement", tilemega::tests::chain_placement_test::TestChainPlacement},
    {"plan_table", tilemega::tests::plan_table_test::TestPlanTable},
    {"cluster_labeling", tilemega::tests::cluster_labeling_test::TestClusterLabeling},
    {"alignment_propagation", tilemega::tests::alignment_propagation_test::TestAlignmentPropagation},
    {"impl_contract", tilemega::tests::impl_contract_test::TestImplContract},
    {"serving_pruning", tilemega::tests::serving_pruning_test::TestServingPruning},
    {"serving_lag", tilemega::tests::serving_lag_test::TestServingLag},
    {"serving_task_index", tilemega::tests::serving_task_index_test::TestServingTaskIndex},
    {"serving_token_sets", tilemega::tests::serving_token_sets_test::TestServingTokenSets},
    {"forward_frontend", tilemega::tests::forward_frontend_test::TestForwardFrontend},
    {"frontend_import", tilemega::tests::frontend_import_test::TestFrontendImport},
    {"dm_descriptor", tilemega::tests::dm_descriptor_test::TestDmDescriptor},
    {"skeleton_placement", tilemega::tests::skeleton_placement_test::TestSkeletonPlacement},
    {"isolated_evaluation", tilemega::tests::isolated_evaluation_test::TestIsolatedEvaluation},
    {"skeleton_search_isolation", tilemega::tests::skeleton_search_isolation_test::TestSkeletonSearchIsolation},
    {"plan_skeleton", tilemega::tests::plan_skeleton_test::TestPlanSkeleton},
    {"variant_resource", tilemega::tests::variant_resource_test::TestVariantResource},
    {"symbolic_oracle", tilemega::tests::symbolic_oracle_test::TestSymbolicOracle},
    {"operator_classes", tilemega::tests::operator_classes_test::TestOperatorClasses},
    {"coupling_cache", tilemega::tests::coupling_cache_test::TestCouplingCache},
    {"semantic_lifting", tilemega::tests::semantic_lifting_test::TestSemanticLifting},
    {"embedding_plan", tilemega::tests::embedding_plan_test::TestEmbeddingPlan},
    {"wiring_coupling", tilemega::tests::wiring_coupling_test::TestWiringCoupling},
    {"frontend_degradation", tilemega::tests::degradation_test::TestDegradation},
    {"model_plan_order_test", tilemega::tests::model_plan_order_test::TestModelPlanOrder},
    {"serving_model_plan_test", tilemega::tests::serving_model_plan_test::TestServingModelPlan},
    {"serving_import_test", tilemega::tests::serving_import_test::TestServingImport},
  };
  if (argc < 2) { std::cerr << "usage: tilemega-unit <case> [args...]\n"; return 2; }
  for (auto const& e : entries)
    if (argv[1] == std::string(e.name)) return e.run(argc - 1, argv + 1);
  std::cerr << "unknown test: " << argv[1] << "\n";
  return 2;
}
