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

#include "zmq_rpc_api.h"
#include <zmq.hpp>
#include <iostream>
#include <stdexcept>

namespace semhelpers {

std::string zmq_rpc_call(const std::string& endpoint,
                         const std::string& request_json,
                         int recv_timeout_ms,
                         int send_timeout_ms) {
    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::req);

    socket.setsockopt(ZMQ_RCVTIMEO, recv_timeout_ms);
    socket.setsockopt(ZMQ_SNDTIMEO, send_timeout_ms);

    socket.connect(endpoint);

    zmq::message_t request(request_json.size());
    memcpy(request.data(), request_json.data(), request_json.size());
    socket.send(request, zmq::send_flags::none);

    zmq::message_t reply;
    if (!socket.recv(reply, zmq::recv_flags::none)) {
        throw std::runtime_error("Timeout: No reply from server");
    }

    return std::string(static_cast<char*>(reply.data()), reply.size());
}

nlohmann::json semantic_filter_zmq_rpc_call(const std::string& name,
                                            const std::vector<std::string>& values,
                                            const std::string& predicate) {
                                                
    const std::string endpoint = "tcp://127.0.0.1:5555";
    nlohmann::json request_json;
    request_json["name"] = name;
    request_json["values"] = values;
    request_json["predicate"] = predicate;

    std::string response_str = zmq_rpc_call(endpoint, request_json.dump());

    nlohmann::json response_json = nlohmann::json::parse(response_str);

    return response_json;
}

nlohmann::json semantic_join_zmq_rpc_call(const std::string& name,
                                            const std::vector<semhelpers::KeyIndexPair>& values,
                                            const std::string& predicate,
                                            const std::string& type) {
                                                
    const std::string endpoint = "tcp://127.0.0.1:5555";
    nlohmann::json request_json;
    request_json["name"] = name;
    request_json["predicate"] = predicate;
    request_json["type"] = type;

    nlohmann::json arr = nlohmann::json::array();
    for (const auto& kv : values) {
      arr.push_back({kv.index, kv.key});
    }
    request_json["values"] = arr;

    std::string response_str = zmq_rpc_call(endpoint, request_json.dump());

    nlohmann::json response_json = nlohmann::json::parse(response_str);

    return response_json;
}

} // namespace semhelpers
