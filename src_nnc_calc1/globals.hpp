// $Id$

/*
Copyright (c) 2007, Trustees of Leland Stanford Junior University
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.
Redistributions in binary form must reproduce the above copyright notice, this 
list of conditions and the following disclaimer in the documentation and/or 
other materials provided with the distribution.
Neither the name of the Stanford University nor the names of its contributors 
may be used to endorse or promote products derived from this software without 
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR 
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _GLOBALS_HPP_
#define _GLOBALS_HPP_
#include <string>

/* kawano */
#include <map>
/* kawano */

/*all declared in main.cpp*/


extern bool _print_activity;

extern int gK;
extern int gN;
extern int gC;


extern int realgk;
extern int realgn;

extern int gNodes;


extern int xrouter;
extern int yrouter;
extern int xcount ;
extern int ycount;

extern bool _trace;

extern bool _use_read_write;

extern double gBurstAlpha;
extern double gBurstBeta;

/*number of flits per packet, set by the configuration file*/
extern int    gConstPacketSize;

extern int *gNodeStates;

extern string watch_file;

/* Matsutani */
extern string trace_file;
extern bool _use_trace_file;
extern int converged_periods;
/* Matsutani */

/* Matsutani EEE */
extern int sleep_threshold;
extern int wakeup_latency;
/* Matsutani EEE */


extern bool _use_noc_latency;

/* Kawano */
extern bool Rold_ejected;
extern int T_reconf;

extern std::map<int, int> reconf_times;
extern int num_rtables;
extern int current_rtable;
extern bool is_reconfroute;
/* Kawano */

#endif
