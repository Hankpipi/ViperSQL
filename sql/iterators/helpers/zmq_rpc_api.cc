#include "zmq_rpc_api.h"
#include <zmq.hpp>
#include <iostream>
#include <stdexcept>

namespace semhelpers {

std::string zmq_rpc_call(const std::string& endpoint,
                         const std::string& request_json,
                         int recv_timeout_ms,
                         int send_timeout_ms) {
    static thread_local zmq::context_t context(1);
    static thread_local zmq::socket_t socket(context, zmq::socket_type::req);
    static thread_local bool connected = false;

    if (!connected) {
        socket.setsockopt(ZMQ_RCVTIMEO, recv_timeout_ms);
        socket.setsockopt(ZMQ_SNDTIMEO, send_timeout_ms);
        socket.connect(endpoint);
        connected = true;
    }

    zmq::message_t request(request_json.size());
    memcpy(request.data(), request_json.data(), request_json.size());
    socket.send(request, zmq::send_flags::none);

    zmq::message_t reply;
    if (!socket.recv(reply, zmq::recv_flags::none)) {
        throw std::runtime_error("Timeout: No reply from server");
    }

    return std::string(static_cast<char*>(reply.data()), reply.size());
}

nlohmann::json semantic_task_zmq_rpc_call(const std::string& name,
                                          const std::vector<std::string>& values,
                                          const std::string& param_key,
                                          const std::string& param_value) {
    const std::string endpoint = "tcp://127.0.0.1:5555";
    nlohmann::json request_json;
    request_json["name"] = name;
    request_json["values"] = values;
    request_json[param_key] = param_value;

    try {
        std::string response_str = zmq_rpc_call(endpoint, request_json.dump());
        return nlohmann::json::parse(response_str);
    } catch (...) {
        return {{"ok", false}, {"error", "zmq or json parse error"}};
    }
}

nlohmann::json semantic_join_zmq_rpc_call(const std::string& name,
                                          const std::vector<semhelpers::KeyIndexPair>& values,
                                          const std::string& predicate,
                                          const std::string& type,
                                          const std::string& join_id) {
    const std::string endpoint = "tcp://127.0.0.1:5555";
    nlohmann::json request_json;
    request_json["name"] = name;
    request_json["predicate"] = predicate;
    request_json["join_type"] = type;
    request_json["join_id"] = join_id;

    nlohmann::json arr = nlohmann::json::array();
    for (const auto& kv : values) {
      arr.push_back({kv.index, kv.key});
    }
    request_json["values"] = arr;

    try {
        std::string response_str = zmq_rpc_call(endpoint, request_json.dump());
        return nlohmann::json::parse(response_str);
    } catch (...) {
        return {{"ok", false}, {"error", "zmq or json parse error"}};
    }
}

} // namespace semhelpers