#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>

#define SEMIHOST_SYS_WRITE 0x05

typedef struct {
  uintptr_t param1;
  uintptr_t param2;
  uintptr_t param3;
} semihostparam_t;

static inline uintptr_t semihost_call(uintptr_t op, uintptr_t arg) {
  register uintptr_t r0 asm("a0") = op;
  register uintptr_t r1 asm("a1") = arg;
  asm volatile(
      ".option push\n"
      ".option norvc\n"
      "slli zero, zero, 0x1f\n"
      "ebreak\n"
      "srai zero, zero, 0x7\n"
      ".option pop\n"
      : "+r"(r0)
      : "r"(r1)
      : "memory");
  return r0;
}

ssize_t _write(int fd, const void* buf, size_t len) {
  (void)fd;
  volatile semihostparam_t arg = {1, (uintptr_t)buf, (uintptr_t)len};
  semihost_call(SEMIHOST_SYS_WRITE, (uintptr_t)&arg);
  return len;
}
