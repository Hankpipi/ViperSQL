#include "sql/sql_sync.h"
#include "mysqld.h"     // 主服务器定义和全局变量
#include "sql_parse.h"  // SQL解析器相关函数
#include "handler.h"    // 表处理器接口


#include <fstream>
#include <iostream> 
#include <chrono>
#include <ctime>
#include <chrono>
#include <thread>

#include <sstream>
#include <iomanip>    // For std::setw and std::left
#include <vector>
#include "sql_class.h"  // THD, Item
#include "query_result.h"
#include "sql_lex.h"
#include "lock.h"


class Query_result; 

class Query_result_logger : public Query_result {
public:
    std::ostringstream result_stream;
    std::vector<std::string> column_names;
    std::vector<size_t> column_widths;

    bool send_result_set_metadata(THD *thd, const mem_root_deque<Item *> &list, uint flags) override {
        (void) thd;
        (void) flags;
        result_stream << "+";
        for (Item *col : list) {
            std::string name = col->item_name.ptr();  // 获取列名
            size_t width = std::max(name.size(), size_t(10));  // 保持最小宽度为10
            column_names.push_back(name);
            column_widths.push_back(width);
            result_stream << std::string(width + 2, '-') << "+";  // 创建表头分隔符
        }
        result_stream << "\n|";

        for (size_t i = 0; i < column_names.size(); ++i) {
            result_stream << " " << std::setw(column_widths[i]) << std::left << column_names[i] << " |";  // 格式化表头
        }
        result_stream << "\n+";
        for (size_t width : column_widths) {
            result_stream << std::string(width + 2, '-') << "+";  // 再次添加分隔符
        }
        result_stream << "\n";
        return false;
    }

    bool send_data(THD *thd, const mem_root_deque<Item *> &items) override {
        (void) thd;
        // result_stream << "|";
        // for (size_t i = 0; i < items.size(); ++i) {
        //     char buff[1024];
        //     String str(buff, sizeof(buff), &my_charset_bin);
        //     if (items[i]->val_str(&str)) {
        //         result_stream << " " << std::setw(column_widths[i]) << std::left << str.c_ptr_safe() << " |";
        //     } else {
        //         result_stream << " " << std::setw(column_widths[i]) << std::left << "NULL" << " |";
        //     }
        // }
        // result_stream << "\n";
        // return false;
        result_stream << "|";
        int i = 0;
        for (auto col : VisibleFields(items)) {
            char buff[1024];
            String str(buff, sizeof(buff), &my_charset_utf8mb4_general_ci);
            String* vstr = col->val_str(&str);
            if(str.length()==1024){
                bool cvstr = col->custom_val_str(&str);
            }                 
            result_stream << " " << std::setw(column_widths[i]) << std::left << str.c_ptr_safe() << " |";
            i++;
        }
        result_stream << "\n";
        return false;
    }

    bool send_eof(THD *thd) override {
        (void) thd;
        result_stream << "+";
        for (size_t width : column_widths) {
            result_stream << std::string(width + 2, '-') << "+";  // 末尾的分隔符
        }
        result_stream << "\n";
        return false;
    }

    void log_reset() {
        result_stream.str("");  // 清空ostringstream
        result_stream.clear();  // 清除任何错误标志
        column_names.clear();   // 清空列名
        column_widths.clear();  // 清空列宽记录
    }

    std::string fetch_results() {
        return result_stream.str();
    }
};


void log_query_result(const std::string& result, const std::string& log_file_path) {
    // 获取当前系统时间
    auto now = std::chrono::system_clock::now();
    std::time_t end_time = std::chrono::system_clock::to_time_t(now);
    
    // 打开日志文件
    std::ofstream log_file(log_file_path, std::ios::app);
    if (log_file.is_open()) {
        // 写入时间戳和查询结果
        log_file << std::ctime(&end_time) << result << std::endl;
        log_file.close();
    } else {
        std::cerr << "Unable to open log file." << std::endl;
    }
}


// std::string fetch_results(Query_result *result) {
//     return "test";
// }
// bool Sql_cmd_sync::execute(THD *thd) { 
//     if (m_orig_cmd) {
//         bool execution_result = m_orig_cmd->execute(thd);
//         std::string s_result = fetch_results(result);       
//         log_query_result(s_result, "../sync.log");
//         return execution_result;
//     }
//     return true;
// }


// bool Sql_cmd_sync::execute(THD *thd) {

//     if (!m_orig_cmd) return true;

//     Query_result_logger logger;
//     Query_result *old_result = thd->lex->result;  // 备份旧的结果处理器
//     thd->lex->result = &logger;  // 设置新的结果处理器

//     for (int i = 0; i < 3; ++i) {
//         logger.log_reset();

//         bool execution_result = m_orig_cmd->execute(thd);
        
//         // std::string s_result = fetch_results(result);  

//         std::string s_result = logger.fetch_results();

//         log_query_result(s_result, "../sync.log");

//         if (execution_result) return true;

//         std::this_thread::sleep_for(std::chrono::seconds(m_interval));

//     }
    
//     thd->lex->result = old_result;  // 恢复旧的结果处理器

//     return false;

// }


bool Sql_cmd_sync::execute(THD *thd) {

    if (!m_orig_cmd) return true; 

    Query_result_logger logger;
    thd->lex->result = &logger;

    for (int i = 0; i < 5; ++i) {
        
        logger.log_reset();
        thd->lex->unit->clear_execution();

        m_orig_cmd->update_bypassed(false); 
        m_orig_cmd->update_prepared(false);

        bool execution_result = m_orig_cmd->execute(thd);
        
        std::string s_result = logger.fetch_results();

        log_query_result(s_result, "../sync.log");

        if (execution_result) return true;

        if (thd->locked_tables_mode) {
            mysql_unlock_tables(thd, thd->lock);
            thd->locked_tables_mode = LTM_NONE;
        }

        std::this_thread::sleep_for(std::chrono::seconds(m_interval));

    }
    
    return false;

}

