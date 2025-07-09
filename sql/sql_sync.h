#ifndef SQL_SYNC_INCLUDED
#define SQL_SYNC_INCLUDED

#include "sql/sql_select.h"

class Sql_cmd_sync : public Sql_cmd_dml {
 public:
  // 构造函数
  Sql_cmd_sync(Sql_cmd_dml* orig_cmd, ulong interval) 
      : m_orig_cmd(orig_cmd), m_interval(interval) {}

  // 重写 execute 方法
  virtual bool execute(THD *thd) override;
  bool precheck(THD *thd) {
    (void) thd;
    return false; 
  }
  bool check_privileges(THD *thd) {
    (void) thd;
    return false; 
  }
  bool prepare_inner(THD *thd) {
    (void) thd;
    return false; 
  }
  enum_sql_command sql_command_code() const { return SQLCOM_SYNC; }


 private:
  Sql_cmd_dml* m_orig_cmd; // 原始 SQL 命令
  ulong m_interval;    // 间隔时间，单位：秒
};





#endif /* SQL_SYNC_INCLUDED */