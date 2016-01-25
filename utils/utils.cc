#ifdef __MACH__
#include <mach/mach.h>
#include <mach/clock.h>
#else
#include <ctime>
#endif

#include "utils.hpp"

double nanoTime() {
#ifdef __MACH__
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  return mts.tv_sec + (mts.tv_nsec / 1E+9);
#else 
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + (t.tv_nsec / 1E+9);
#endif
};


