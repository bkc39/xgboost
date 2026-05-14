/* libxgbshim.so — companion library bundled with libxgbcompat.so on Linux.
 *
 * Provides definitions for glibc symbols that polyfill-glibc cannot rewrite
 * down to an older glibc baseline.  By redirecting the bundled libs to
 * resolve these symbols here (via --rename-dynamic-symbols), the package
 * loads cleanly on hosts whose system libc/libm are older than the Nix
 * toolchain's (e.g. pkg-build.racket-lang.org).
 *
 *   __libc_single_threaded  (glibc 2.32+, libc): a uint8_t hint.  We export
 *       a zero byte ("not known single-threaded"), which forces any caller
 *       to take the thread-safe path — always behaviorally correct.
 *
 *   strfromf128 / strtof128 (glibc 2.26+, libc): _Float128 <-> string.
 *       Pulled in by libstdc++ for std::to_chars/from_chars on __float128
 *       but not reached by xgboost.  Stubs abort if ever called.
 *
 *   getentropy              (glibc 2.25+, libc): used by libstdc++'s
 *       std::random_device default constructor.  xgboost doesn't construct
 *       one, so we stub.
 */

#include <stddef.h>
#include <stdlib.h>

__attribute__((visibility("default"))) const char __libc_single_threaded = 0;

__attribute__((visibility("default")))
int strfromf128(char *dest, size_t size, const char *format, long double f) {
    (void)dest; (void)size; (void)format; (void)f;
    abort();
}

__attribute__((visibility("default")))
long double strtof128(const char *nptr, char **endptr) {
    (void)nptr; (void)endptr;
    abort();
}

__attribute__((visibility("default")))
int getentropy(void *buffer, size_t length) {
    (void)buffer; (void)length;
    abort();
}
