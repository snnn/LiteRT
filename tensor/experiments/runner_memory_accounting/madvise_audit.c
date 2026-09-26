/*
 * Copyright 2026 Google LLC.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdatomic.h>
#include <stdio.h>
#include <sys/mman.h>
static _Atomic unsigned long pageout_calls, pageout_bytes;
int madvise(void* address, size_t size, int advice) {
  int (*next)(void*, size_t, int) = dlsym(RTLD_NEXT, "madvise");
  if (advice == 21) {
    atomic_fetch_add(&pageout_calls, 1);
    atomic_fetch_add(&pageout_bytes, size);
  }
  return next(address, size, advice);
}
__attribute__((destructor)) static void report(void) {
  fprintf(stderr, "PAGEOUT_AUDIT calls=%lu requested_bytes=%lu\n",
          atomic_load(&pageout_calls), atomic_load(&pageout_bytes));
}
