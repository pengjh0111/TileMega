// V-J intentionally reuses the exact V-A synchronization harness.  The
// canonical source contains the kBackward dependency added for this experiment
// and command-line tile sizing; including it here keeps V-J reproducible
// without maintaining a divergent copy.
#include "../V_A/event_sync.cu"
