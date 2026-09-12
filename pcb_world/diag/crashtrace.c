/* Fatal-signal tracer — compiled and dlopen'd by pcb_world/diag/
 * crash_handler.py (pytest and training/eval processes alike). On SIGSEGV/
 * SIGABRT/SIGBUS/SIGILL/SIGFPE it appends the native backtrace to
 * $CRASHTRACE_FILE, then
 * re-delivers the signal to the handler installed before us (Python
 * faulthandler), so the same file also gets the Python stack. glibc >= 2.35
 * removed libSegFault.so; this is the minimal replacement.
 * Build: gcc -shared -fPIC -O1 -o <out>.so crashtrace.c
 */
#define _GNU_SOURCE
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static struct sigaction prev[NSIG];

static void handler(int sig) {
    const char *fname = getenv("CRASHTRACE_FILE");
    if (fname) {
        int fd = open(fname, O_CREAT | O_WRONLY | O_APPEND, 0644);
        if (fd >= 0) {
            char head[128];
            int n = snprintf(head, sizeof(head),
                             "=== fatal signal %d (%s) pid %d — native stack ===\n",
                             sig, strsignal(sig), (int)getpid());
            if (write(fd, head, n) < 0) { /* best effort */ }
            void *buf[64];
            backtrace_symbols_fd(buf, backtrace(buf, 64), fd);
            close(fd);
        }
    }
    /* re-deliver to the previous handler (faulthandler -> Python stack, then
     * it re-raises the default action); fall back to default if none/ignored */
    sigaction(sig, &prev[sig], NULL);
    if (!(prev[sig].sa_flags & SA_SIGINFO) && prev[sig].sa_handler == SIG_IGN)
        signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor)) static void install(void) {
    void *buf[4];
    backtrace(buf, 4); /* force-load libgcc now so backtrace() is handler-safe */
    int sigs[] = {SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE};
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handler;
    sigemptyset(&sa.sa_mask);
    for (unsigned i = 0; i < sizeof(sigs) / sizeof(sigs[0]); i++)
        sigaction(sigs[i], &sa, &prev[sigs[i]]);
}
