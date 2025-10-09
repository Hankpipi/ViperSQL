#ifndef SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_ZMQ_RPC_H_
#define SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_ZMQ_RPC_H_

/*
   Copyright (c) 2025, Songsong Mo

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; version 2 of the License.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the
   Free Software Foundation, Inc., 59 Temple Place, Suite 330,
   Boston, MA  02111-1307  USA
*/


#include <string>
#include <nlohmann/json.hpp>

namespace zmq { class context_t; }

namespace semhelpers {

/**
 * zmq_rpc_call
 * Send a JSON request to a ZeroMQ REP server and string response.
 * endpoint: "ipc:///tmp/sem.sock" or "tcp://127.0.0.1:5555"
 */
std::string zmq_rpc_call(const std::string& endpoint,
                         const std::string& request_json,
                         int recv_timeout_ms = 5000,
                         int send_timeout_ms = 5000);
      
/**
 * semantic_filter_zmq_rpc_call
 * input:  filter_name: "sem_filter", values:[...], predicate: "..."
 * output: { "name":"sem_filter", "values":[...] }
 */
nlohmann::json semantic_filter_zmq_rpc_call(const std::string& name,
                                            const std::vector<std::string>& values,
                                            const std::string& predicate);

/**
 * semantic_join_zmq_rpc_call
 * input:  join_name: "sem_join", values:[...], predicate: "...", type: "build"
 * output: none / { "name":"sem_filter", "values":[...] }
 */                                           
nlohmann::json semantic_join_zmq_rpc_call(const std::string& name,
                                            const std::vector<std::string>& values,
                                            const std::string& predicate,
                                            const std::string& type)

} // namespace semhelpers

#endif // SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_ZMQ_RPC_H_
