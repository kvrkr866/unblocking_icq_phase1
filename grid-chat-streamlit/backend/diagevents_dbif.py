"""
##########################################################
#
# Capstone Team 16
#
# Author: RK (kvrkr866@gmail.com)
#
#  diagevents SQLite3 DB interface module
#     The purpose of this module is to execute the query on griddiagnostics DB
#        and sends the results back to the caller. 
#     The user request will be passed to LLM to prepare the SQL query, the same query
#      will be send as input this module.  
#
#     Please refer the following files for Unit test - 
#         1) tests\griddiagnostics_ut_test.py
#         2) tests\griddiagnostics_llm_test.ipynb
#
"""

import sqlite3
import datetime

from constants import *


#######################################################################
#### Tables Schema details JSON format - START ########################
"""
{
  "severity": {
    "severity_id": "INTEGER PRIMARY KEY",
    "severity_name": "TEXT NOT NULL",
    "severity_desc": "TEXT"
  },
  "event": {
    "event_id": "INTEGER PRIMARY KEY",
    "event_name": "TEXT NOT NULL",
    "event_type": "TEXT NOT NULL",
    "event_desc": "TEXT"
  },
  "eventslog": {
    "event_id": "INTEGER (FOREIGN KEY to event.event_id)",
    "severity_id": "INTEGER (FOREIGN KEY to severity.severity_id)",
    "timestamp": "DATETIME",
    "status": "TEXT"
  }
}
"""
#### Tables Schema details JSON format - END ########################


#### Tables Schema details Suitetable in LLM promot format - START ####
"""
The database contains three tables with these schemas:

1. severity(severity_id INTEGER PRIMARY KEY, severity_name TEXT NOT NULL, severity_desc TEXT)
2. event(event_id INTEGER PRIMARY KEY, event_name TEXT NOT NULL, event_type TEXT NOT NULL, event_desc TEXT)
3. eventslog(event_id INTEGER, severity_id INTEGER, timestamp DATETIME, status TEXT, foreign keys to event and severity)

Write SQL queries based on this schema...
"""
#### Tables Schema details Suitetable in LLM promot format - END ####
#######################################################################

##########################################################
def gd_userquery_execute(userquery: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(userquery)
    query_response = c.fetchall()
    conn.close()
    return query_response

##########################################################
# Usage example: create the tables in 'events_database.db'
if __name__ == "__main__":
    print(" ... STARTED ....  ")
    print(" DumpUserQuery RESULTS  START ")
    query="""
SELECT e.event_id,
       e.event_name,
       e.event_type,
       s.severity_id,
       s.severity_name,
       l.timestamp,
       l.status
FROM eventslog l
JOIN event e ON l.event_id = e.event_id
LEFT JOIN severity s ON l.severity_id = s.severity_id
WHERE l.timestamp >= datetime('now', '-1 day')
ORDER BY l.timestamp DESC;
    """
    userquery_results = gd_userquery_execute(query)
    for row in userquery_results:
        print(row)
    print(" DumpUserQuery RESULTS  END ")

    print(" ... END ....  ")

