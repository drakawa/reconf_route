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

#ifndef _ROUTEFUNC_HPP_
#define _ROUTEFUNC_HPP_

#include "flit.hpp"
#include "router.hpp"
#include "outputset.hpp"
#include "config_utils.hpp"

typedef void (*tRoutingFunction)( const Router *, const Flit *, int in_channel, OutputSet *, bool );

void InitializeRoutingMap( );
int fattree_transformation(int dest);
tRoutingFunction GetRoutingFunction( const Configuration& config );

extern map<string, tRoutingFunction> gRoutingFunctionMap;
extern int gNumVCS;
extern int gReadReqBeginVC, gReadReqEndVC;
extern int gWriteReqBeginVC, gWriteReqEndVC;
extern int gReadReplyBeginVC, gReadReplyEndVC;
extern int gWriteReplyBeginVC, gWriteReplyEndVC;
extern int memo_log2gC ;

/* kawano */
extern map<int, map<int, int>> global_routing_table; // grt[curr][dst] = out-port
extern map<int, map<int, int>> global_routing_table_vc; // grt[curr][dst] = vc-out
extern map<int, map<int, vector<tuple<int, int, int>>>> global_routing_table_nvp;
extern map<int, map<int, vector<tuple<int, int, int, int, int>>>> global_routing_table_ionvp; // nvp[curr][dst] = {(in-port,in-vc,out-port,out-vc,pri), ...}

extern map<int, map<int, int>> next_node_port; // nnp[curr]={next:out-port, next2:out2, ...}
extern map<int, map<int, int>> prev_node_port; // pnp[curr]={prev:in-port, prev2:in-port2, ...}
extern map<int, map<int, int>> next_port_node; // npn[curr]={out-port:next, out2:next2, ...}
extern map<int, map<int, int>> prev_port_node; // ppn[curr]={in-port:prev,  in2:prev2, ...}
/* kawano */

extern int _use_vc;
/* ozaki */

#endif
