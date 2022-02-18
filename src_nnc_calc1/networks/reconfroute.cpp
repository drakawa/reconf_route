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

/*reconfroute
 *
 *The 
 *
 */

#include "reconfroute.hpp"
#include <fstream>
#include <sstream>
#include <limits>
#include <algorithm>

#include <map>
using namespace std;

//this is a hack, I can't easily get the routing talbe out of the network

ReconfRoute::ReconfRoute( const Configuration &config )
    :  Network( config ){

    _ComputeSize( config );
    _Alloc( );
    _BuildNet( config );
  
}


void ReconfRoute::_ComputeSize( const Configuration &config ){
    config.GetStr("network_file",file_name);
    /* ozaki */
    config.GetStr("routing_table_file",txtfile_name);

    _use_vc = config.GetInt("use_vc");

    if(file_name==""){
	cout<<"No network file name provided"<<endl;
	exit(-1);
    }
    if(txtfile_name==""){
	cout<<"No routing table file name provided"<<endl;
	exit(-1);
    }
    //parse the network description file
    readFile();

    _channels =0;
    cout<<"========================Network File Parsed=================\n";
    cout<<"******************node listing**********************\n";
    map<int,  int >::const_iterator iter;
    for(iter = node_list.begin(); iter!=node_list.end(); iter++){
	cout<<"Node "<<iter->first;
	cout<<"\t Router "<<iter->second<<endl;
    }

    map<int,   map<int, int >*>::const_iterator iter3;
    cout<<"\n****************router to node listing*************\n";
    for(iter3 = router_list[0].begin(); iter3!=router_list[0].end(); iter3++){
	cout<<"Router "<<iter3->first<<endl;
	map<int, int >::const_iterator iter2;
	for(iter2 = iter3->second->begin(); iter2!=iter3->second->end(); iter2++){
	    cout<<"\t Node "<<iter2->first<<endl;
	}
    }

    cout<<"\n*****************router to router listing************\n";
    for(iter3 = router_list[1].begin(); iter3!=router_list[1].end(); iter3++){
	cout<<"Router "<<iter3->first<<endl;
	map<int, int >::const_iterator iter2;
	if(iter3->second->size() == 0){
	    cout<<"Caution Router "<<iter3->first
		<<" is not connected to any other Router\n"<<endl;
	}
	for(iter2 = iter3->second->begin(); iter2!=iter3->second->end(); iter2++){
	    cout<<"\t Router "<<iter2->first<<endl;
	    _channels++;
	}
    }

    _size = router_list[1].size();
    _sources = node_list.size();
    _dests = _sources;
    //routing_table = new map<int, int>[_size];


}



void ReconfRoute::_BuildNet( const Configuration &config ){
  

    //I need to keep track the output ports for each router during build
    int * outport = (int*)malloc(sizeof(int)*_size);
    for(int i = 0; i<_size; i++){outport[i] = 0;}
    /* kawano */
    int * inport = (int *)malloc(sizeof(int)*_size);
    for(int i = 0; i<_size; i++){inport[i] = 0;}
    /* kawano */

    cout<<"==========================Node to Router =====================\n";
    //adding the injection/ejection chanenls first
    map<int,   map<int, int >*>::const_iterator niter;
    for(niter = router_list[0].begin(); niter!=router_list[0].end(); niter++){
	map<int,   map<int, int >*>::const_iterator riter = router_list[1].find(niter->first);
	//calculate radix
	int radix = niter->second->size()+riter->second->size();
	int node = niter->first;
	cout<<"router "<<node<<" radix "<<radix<<endl;
	//decalre the routers 
	ostringstream router_name;
	router_name << "router";
	router_name << "_" <<  node ;
	_routers[node] = Router::NewRouter( config, this, router_name.str( ), 
					    node, radix, radix );
	//add injeciton ejection channels
	map<int, int >::const_iterator nniter;
	for(nniter = niter->second->begin();nniter!=niter->second->end(); nniter++){
	    int link = nniter->first;
	    //add the outport port assined to the map
	    (*(niter->second))[link] = outport[node];

	    /* kawano */
	    next_node_port[node][node] = outport[node]; // (curr, curr)-> out-port=0
	    prev_node_port[node][node] = inport[node]; // (curr, curr)-> in-port=0

	    next_port_node[node][outport[node]] = node; // (curr, 0) -> out-node=curr
	    prev_port_node[node][inport[node]] = node; // (curr, 0) -> in-node=curr
	    
	    inport[node]++;
	    outport[node]++;
	    /* kawano */

	    cout<<"\t connected to node "<<link<<" at outport "<<nniter->second<<endl;
	    _routers[node]->AddInputChannel( &_inject[link], &_inject_cred[link] );
	    _routers[node]->AddOutputChannel( &_eject[link], &_eject_cred[link] );
	}

    }

    cout<<"==========================Router to Router =====================\n";
    //add inter router channels
    //since there is no way to systematically number the channels we just start from 0
    //the map, is a mapping of output->input
    int channel_count = 0; 
    for(niter = router_list[0].begin(); niter!=router_list[0].end(); niter++){
	map<int,   map<int, int >*>::const_iterator riter = router_list[1].find(niter->first);
	int node = niter->first;
	map<int, int >::const_iterator rriter;
	cout<<"router "<<node<<endl;
	for(rriter = riter->second->begin();rriter!=riter->second->end(); rriter++){
	    int other_node = rriter->first;
	    int link = channel_count;
	    //add the outport port assined to the map
	    (*(riter->second))[other_node] = outport[node];
	    /* kawano */
	    next_node_port[node][other_node] = outport[node]; // (curr, next)-> out-port
	    prev_node_port[other_node][node] = inport[other_node]; // (curr, prev)-> in-port

	    next_port_node[node][outport[node]] = other_node;
	    prev_port_node[other_node][inport[other_node]] = node;

	    outport[node]++;
	    inport[other_node]++;
	    /* kawano */

	    cout<<"\t connected to router "<<other_node<<" using link "<<link<<" at outport "<<rriter->second<<endl;
	    _routers[node]->AddOutputChannel( &_chan[link], &_chan_cred[link] );
	    _routers[other_node]->AddInputChannel( &_chan[link], &_chan_cred[link]);
	    channel_count++;
	}
    }

    buildRoutingTable();

}


void ReconfRoute::RegisterRoutingFunctions() {
    gRoutingFunctionMap["min_reconfroute"] = &min_reconfroute;
}

void min_reconfroute( const Router *r, const Flit *f, int in_channel, 
		 OutputSet *outputs, bool inject ){
    outputs->Clear( );
    int out_port = -1;
    int rID = r->GetID();
    int dest = f->dest;
  
    //out_port = global_routing_table[rID].find(dest)->second;

    int vcBegin = 0, vcEnd = gNumVCS-1;
    // kawano
    int vc_pri = 0;
	int in_port, in_vc;

    if ( f->type == Flit::READ_REQUEST ) {
	vcBegin = gReadReqBeginVC;
	vcEnd   = gReadReqEndVC;
	outputs->AddRange(out_port, vcBegin, vcEnd);
    } else if ( f->type == Flit::WRITE_REQUEST ) {
	vcBegin = gWriteReqBeginVC;
	vcEnd   = gWriteReqEndVC;
	outputs->AddRange(out_port, vcBegin, vcEnd);
    } else if ( f->type ==  Flit::READ_REPLY ) {
	vcBegin = gReadReplyBeginVC;
	vcEnd   = gReadReplyEndVC;
	outputs->AddRange(out_port, vcBegin, vcEnd);
    } else if ( f->type ==  Flit::WRITE_REPLY ) {
	vcBegin = gWriteReplyBeginVC;
	vcEnd   = gWriteReplyEndVC;
	outputs->AddRange(out_port, vcBegin, vcEnd);
    } else if ( f->type ==  Flit::ANY_TYPE ) {
	//vcBegin = 0;
	//vcEnd = gNumVCS-1;

	/* kawano */
	if(!_use_vc){
	    out_port = global_routing_table[rID][dest];
	    vcBegin = 0;
	    vcEnd = gNumVCS-1;
		vc_pri = 0;
	    outputs->AddRange(out_port, vcBegin, vcEnd, vc_pri);
	} else if(_use_vc){
	    if (rID == dest) { // eject
		out_port = 0;
		vcBegin = 0;
		vcEnd = gNumVCS-1;
		vc_pri = 0;
	    outputs->AddRange(out_port, vcBegin, vcEnd, vc_pri);

	    // } else if (in_channel == 0) { // inject
		// out_port = global_routing_table[rID][dest];
		// vcBegin = 0;
		// vcEnd   = 0;
		// vc_pri = 0;
	    // outputs->AddRange( out_port , vcBegin, vcEnd );
	    } else {
			vector<tuple<int, int, int, int, int>> tmp_grt_ionvp_vec = global_routing_tables_ionvp_vec[f->rtable_idx][rID][dest];
			for (size_t i = 0; i < tmp_grt_ionvp_vec.size(); i++) {
				in_port =     get<0>(tmp_grt_ionvp_vec[i]);
				in_vc =       get<1>(tmp_grt_ionvp_vec[i]);
				out_port =    get<2>(tmp_grt_ionvp_vec[i]);
				vcBegin  =    get<3>(tmp_grt_ionvp_vec[i]);
				vcEnd    =    get<3>(tmp_grt_ionvp_vec[i]);
				vc_pri   =    get<4>(tmp_grt_ionvp_vec[i]);
				if (in_channel == in_port && f->vc == in_vc) {
					outputs->AddRange(out_port, vcBegin, vcEnd, vc_pri);
				}
			}
		// cout << global_routing_tables_ionvp_vec[current_rtable][rID][dest].size() << endl;
		// cout << "src= " << rID << " dst= " << dest << " out_port= " << out_port << " (next= " << next_port_node[rID][out_port] << ") to " << vcBegin << " w/ priority " << vc_pri << endl;
	    }
	    /* kawano */

		// cout << "src= " << rID << " dst= " << dest << " vc from " << f->vc << " to " << vcBegin << endl;
	    if (vcBegin > gNumVCS - 1) {
		cout << "Invalid_vc: src= " << rID << " dst= " << dest << " vc from " << f->vc << " to " << vcBegin << endl;
	    }

	}
    }

}

void ReconfRoute::buildRoutingTable(){
    cout<<"==========================Router to Router =====================\n";  

	cout << "txtfile file:" << txtfile_name << endl;
    ifstream ifs_txt(txtfile_name.c_str());
	string str_txt;
	string rfile_name;
	char rfile_name_char[100];
	int time_reconf;

	int txtfile_countline = 0;

	while (getline(ifs_txt, str_txt)) {
		sscanf(str_txt.data(), "%s %d", rfile_name_char, &time_reconf);
		rfile_name = rfile_name_char;

		map<int, map<int, vector<tuple<int, int, int, int, int>>>> tmp_grtable_ionvp;

		ifstream ifs_r(rfile_name.c_str());

		string str;
		int pn, pv, s, d, n, nvc, v, p;

		cout << "routing table file:" << rfile_name << " with reconf time=" << time_reconf << endl;
		if(!_use_vc){
		while(getline(ifs_r, str)){
			/* kawano */
			sscanf(str.data(), "%d %d %d %d", &s, &d, &n, &nvc);
			global_routing_table[s][d] = next_node_port[s][n];
			global_routing_table_vc[s][d] = nvc;
		}
	
		}
		else if(_use_vc){
		while(getline(ifs_r, str)){
			/* kawano */
			sscanf(str.data(), "%d %d %d %d %d %d %d", &pn, &pv, &s, &d, &n, &v, &p);
			int prev_port = prev_node_port[s][pn];
			int next_port = next_node_port[s][n];
			tuple<int, int, int, int, int> tmp_t = make_tuple(prev_port, pv, next_port, v, p);
			// cout << prev_port << " " << pv << " " << next_port << " " << v << " " << p << endl;
			tmp_grtable_ionvp[s][d].push_back(tmp_t);
		}
		global_routing_tables_ionvp_vec.push_back(tmp_grtable_ionvp);
		reconf_times[txtfile_countline] = time_reconf;
		txtfile_countline++;
		num_rtables++;
		// cout << "ionvp_vc size: " << global_routing_tables_ionvp_vec[0].size() << endl;
		}
		cout << endl;
	}
}

int ReconfRoute::findPath(int router, int dest, int* hop_count,map<int, bool>* visited){

    /* Hard case
     * hop hop hop
     */
    //alright been here, aint no pather this way
    // cout<<"\t\t*at router "<<router<<endl;
    if(visited->find(router)!= visited->end()){
	//   cout<<"\t\t\t*running in a circle"<<endl;
	return -1;
    }

    if((*((router_list[0]).find(router)->second)).find(dest)!=(*((router_list[0]).find(router)->second)).end()){
    
	//cout<<"\t\t\t*found node returning"<<endl;
	return (*((router_list[0]).find(router)->second)).find(dest)->second;
    }
  
    (*visited)[router] = true;
  
    map<int,   map<int, int >*>::const_iterator riter = router_list[1].find(router);
    map<int, int >::const_iterator rriter;

    int shortest_distance = 99999;
    int shortest_port = -1;
    for(rriter = riter->second->begin();rriter!=riter->second->end(); rriter++){
	int outport = -1;
	int duplicate_hop_count = *hop_count;
	outport = findPath(rriter->first,dest,&duplicate_hop_count, visited);
	//omg we found a path??
	if(outport !=-1){
	    if(duplicate_hop_count<shortest_distance){
		shortest_distance = duplicate_hop_count;
		shortest_port = rriter->second;
	    }

	}
    }
    visited->erase(visited->find(router));
    (*hop_count) = shortest_distance+1;
    return shortest_port;
}


void ReconfRoute::readFile(){

    ifstream network_list;
    string line;
    router_list = new map<int,  map<int, int>* >[2];

    network_list.open(file_name.c_str());
    if(!network_list.is_open()){
	cout<<"can't open network file "<<file_name<<endl;
	exit(-1);
    }
  
    //loop through the entire file
    while(!network_list.eof()){
	getline(network_list,line);
	if(line==""){
	    continue;
	}
	//position to parse out white sspace
	int pos = 0;
	int next_pos=-1;
	//the first node and its type, 0 node, 1 router
	bool head = false;
	int head_position = -1;
	int head_type = -1;
	//rest of the connections
	bool name = false;
	int body_type = -1;

	//loop through each element in a line
	do{
	    next_pos = line.find(" ",pos);
	    string temp = line.substr(pos,next_pos-pos);
	    pos = next_pos+1;
	    //skip empty spaces
	    if(temp=="" || temp==" "){
		continue;
	    }

	    //////////////////////////////////
	    //real parsing begins
	    if(name){
		int id = atoi(temp.c_str());
		//if this is a head 
		if(!head){ //indicates the beginin of the line
		    head = true;
		    head_type = body_type;
		    head_position = id;
		} else { 
		    //if this is a body parse, approriately
		    if(body_type==-1){
			cout<<"illegal body type\n";
			exit(-1);
		    }
		    //use map depending on head type
		    switch(head_type){
		    case 0: //node
			if(body_type==0){
			    cout<<"can not connect node to node\n";
			    exit(-1);
			}
			//insert into the node list
			if(node_list.find(head_position) != node_list.end()){
			    if(node_list.find(head_position)->second!=id){
				cout<<"node "<<head_position<<" is conncted to multiple routers! "
				    <<id<<" and "<<node_list.find(head_position)->second<<endl;
				exit(-1);
			    }
			} else {
			    node_list[head_position]=id;
			}
			//reverse insert
			if(router_list[0].find(id) == router_list[0].end()){
			    (router_list[0])[id] = new map<int, int>;
			} 
			(*((router_list[0]).find(id)->second))[head_position]=-1;
			//initialize the other router array as well
			if(router_list[1].find(id) == router_list[1].end()){
			    (router_list[1])[id] = new map<int, int>;
			} 
			break;
		    case 1: //router
			//insert into router list
			if(router_list[body_type].find(head_position) == router_list[body_type].end()){
			    (router_list[body_type])[head_position] = new map<int, int>;
			} 
			(*((router_list[body_type]).find(head_position)->second))[id]=-1;

			//reverse insert
			if(body_type == 0){
			    if(node_list.find(id) == node_list.end()){
				node_list[id] = head_position;
			    }  else {
				if(node_list.find(id)->second!=head_position){
				    cout<<"node "<<id<<" is conncted to multiple routers! "
					<<head_position<<" and "<<node_list.find(id)->second<<endl;
				    exit(-1);
				}
			    }
			    //initialize the other array as well
			    if(router_list[1].find(head_position) == router_list[1].end()){
				(router_list[1])[head_position] = new map<int, int>;
			    } 
			} else {
			    if(router_list[1].find(id) == router_list[1].end()){
				(router_list[1])[id] = new map<int, int>;
			    } 
			    (*((router_list[1]).find(id)->second))[head_position]=-1;
			    //initialize the other array as well
			    if(router_list[0].find(head_position) == router_list[0].end()){
				(router_list[0])[head_position] = new map<int, int>;
			    } 
			    if(router_list[0].find(id) == router_list[0].end()){
				(router_list[0])[id] = new map<int, int>;
			    } 
			}

			break;
		    default:
			cout<<"illegal head type\n";
			exit(-1);
		    }
		}
		name = false;
		body_type = -1;
	    } else {	//setting type name
		name= true;
		if(temp=="router"){
		    body_type = 1;
		} else if (temp == "node"){
		    body_type = 0;
		}
	    }
	} while(pos!=0);

    }

 

    //map verification, make sure the information contained in bother maps
    //are the same
    assert(router_list[0].size() == router_list[1].size());

}

