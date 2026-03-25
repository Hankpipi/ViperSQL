#ifndef SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_ZMQ_RPC_H_
#define SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_ZMQ_RPC_H_

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "sql/iterators/helpers/sem_join_helper.h"

namespace zmq { class context_t; }

namespace semhelpers {

std::string zmq_rpc_call(const std::string& endpoint,
                         const std::string& request_json,
                         int recv_timeout_ms = 120000,
                         int send_timeout_ms = 5000);
      
/**
 * semantic_task_zmq_rpc_call
 * A unified generic RPC caller for tasks that send a 1D array of strings.
 * param_key can be "predicate" or "instruction".
 */
nlohmann::json semantic_task_zmq_rpc_call(const std::string& name,
                                          const std::vector<std::string>& values,
                                          const std::string& param_key,
                                          const std::string& param_value);
                                        
nlohmann::json semantic_join_zmq_rpc_call(const std::string& name,
                                          const std::vector<semhelpers::KeyIndexPair>& values,
                                          const std::string& predicate,
                                          const std::string& type,
                                          const std::string& join_id);

} // namespace semhelpers

#endif // SQL_ITERATORS_EXTERNAL_HELPERS_SEMHELPERS_ZMQ_RPC_H_