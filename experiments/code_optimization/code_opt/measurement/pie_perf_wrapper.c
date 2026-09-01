/*
 * SIGSTOP-then-exec wrapper for the perf backend.
 *
 * Eliminates the perf-attach race window. Flow:
 *   1. Parent calls Popen([wrapper, argv1, argv2, ...]).
 *   2. Kernel forks + execs the wrapper. The wrapper's first action is
 *      raise(SIGSTOP), which suspends the process.
 *   3. Parent uses waitpid(pid, WUNTRACED) to detect WIFSTOPPED.
 *   4. Parent calls perf_event_open(pid) — child is stopped, no work yet.
 *   5. Parent calls kill(pid, SIGCONT) — child resumes.
 *   6. Wrapper executes execvp(argv[1], &argv[1]).
 *      perf event survives execve (kernel preserves per-PID counters).
 *   7. Real candidate program runs with perf attached from instruction 0.
 *
 * Compile once at module init via gcc; no runtime deps.
 *
 * On execvp failure (target binary missing/non-executable) the wrapper
 * exits with 127 — same convention as /bin/sh.
 */
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "pie_perf_wrapper: missing target\n");
        return 127;
    }
    /* Suspend self so the parent can attach perf_event_open before any
     * candidate-program instructions execute. */
    if (raise(SIGSTOP) != 0) {
        perror("pie_perf_wrapper: raise(SIGSTOP)");
        return 127;
    }
    /* Resume here after parent SIGCONTs. Drop the wrapper out of the
     * picture via execvp; perf is now counting everything from the real
     * binary's first instruction. */
    execvp(argv[1], &argv[1]);
    perror("pie_perf_wrapper: execvp");
    return 127;
}
