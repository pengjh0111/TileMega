# Window timing-probe coverage repair

Source inspection during the first ablation session found that
`TILEMEGA_UNSAFE_NO_EVENT_WAIT` bypassed WaitTaskDependencies but not
ProbeTaskDependencies. At W>1 the nowait/neither arms therefore retained
global event polling and the CTA-wide probe reduction. The initial matrix
was stopped before any required cell reached 25 complete rounds and retained
under `../ablation_pre_probe_fix/`; none of it is used for final conclusions.
The B prerequisite and all W=1 probes are unaffected.

The repair returns ready immediately from ProbeTaskDependencies in the
unsafe no-wait build. It changes no safe execution path. The two complete
W=2 safe SASS dumps are byte-identical after recompiling with the same source
paths. The first control compilation used a different source path, producing
only three `identifier` header differences; it was rebuilt with the original
source path, rather than normalizing or deleting lines from the SASS proof.

The gqa2 W=2 nowait L2 kernel loses its one BAR.RED and one event-polling
ATOMG site. `before.sass`, `after.sass`, `proof.json`, the saved original
build metadata and `verification.log` preserve the evidence. The current
32 reference-window and four real-width-window unsafe binaries were rebuilt;
full/nofence/l1nosync images and their correctness evidence are unchanged.

Independent Chain2 and fixed-candidate measurements proceed while the probe
is rebuilt. The final complete protocol matrix is restarted as one new
session across all seven configurations and both placements. The research
formula, sample count, confidence calculation and coverage are unchanged.
This repairs the implementation of the no-wait arm, not the acceptance gate.
