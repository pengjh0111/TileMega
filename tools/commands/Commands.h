// SPDX-License-Identifier: BSD-3-Clause
#pragma once
namespace tilemega::commands::attention_work { int RunAttentionWork(int, char**); }
namespace tilemega::commands::calibrate { int RunCalibrate(int, char**); }
namespace tilemega::commands::compile { int RunCompile(int, char**); }
namespace tilemega::commands::derive { int RunDerive(int, char**); }
namespace tilemega::commands::dram_floor { int RunDramFloor(int, char**); }
namespace tilemega::commands::event_cost { int RunEventCost(int, char**); }
namespace tilemega::commands::flow_audit { int RunFlowAudit(int, char**); }
namespace tilemega::commands::import { int RunImport(int, char**); }
namespace tilemega::commands::interface_probe { int RunInterfaceProbe(int, char**); }
namespace tilemega::commands::occupancy { int RunOccupancy(int, char**); }
namespace tilemega::commands::op_audit { int RunOpAudit(int, char**); }
namespace tilemega::commands::parametric { int RunParametric(int, char**); }
namespace tilemega::commands::runtime_projection { int RunRuntimeProjection(int, char**); }
namespace tilemega::commands::scalar_error_probe { int RunScalarErrorProbe(int, char**); }
namespace tilemega::commands::skeleton_audit { int RunSkeletonAudit(int, char**); }
namespace tilemega::commands::target_audit { int RunTargetAudit(int, char**); }
namespace tilemega::commands::task_work_probe { int RunTaskWorkProbe(int, char**); }
namespace tilemega::commands::wait_policy { int RunWaitPolicy(int, char**); }
namespace tilemega::commands::wiring { int RunWiring(int, char**); }
namespace tilemega::commands::audit_binary { int RunSass(int, char**); int RunArch(int, char**); }
namespace tilemega::commands::device { int RunDevice(int, char**); }
namespace tilemega::commands::calibration_suite { int RunSuite(int, char**); }
