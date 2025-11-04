##########################################################
#
# Capstone Team 16
# Utility program to check the RDBMS operations on sqlite3
#   This was implemented as part grid diagnostics information retrieval 
#     Diagnostics information will be communicated in the form of events.
#

import sqlite3
#from pydantic import BaseModel
import datetime

DB_NAME = 'gridevents.db'
TABLE_EVENT = 'event'
TABLE_SEVERITY  = 'severity'
TABLE_EVENTSLOG = 'eventslog'

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
def create_tables(db_name: str):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    # Create severity table
    c.execute('''
        CREATE TABLE IF NOT EXISTS severity (
            severity_id INTEGER PRIMARY KEY,
            severity_name TEXT NOT NULL,
            severity_desc TEXT
        )
    ''')
    
    # Create event table
    c.execute('''
        CREATE TABLE IF NOT EXISTS event (
            event_id INTEGER PRIMARY KEY,
            event_name TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_desc TEXT
        )
    ''')
    
    # Create eventslog table
    c.execute('''
        CREATE TABLE IF NOT EXISTS eventslog (
            event_id INTEGER,
            severity_id INTEGER,
            timestamp DATETIME,
            status TEXT,
            FOREIGN KEY (event_id) REFERENCES event(event_id),
            FOREIGN KEY (severity_id) REFERENCES severity(severity_id)
        )
    ''')
    
    conn.commit()
    conn.close()

##########################################################
def insert_data_into_tables(db_name: str):

    # Insert sample severity data
    severities = [
        (1, 'Low', 'Low severity issue'),
        (2, 'Medium', 'Medium severity issue'),
        (3, 'High', 'High severity issue')
    ]


    # Insert sample event data
    events = [
        (1, 'Disk Space Low', 'System', 'Disk space is running low on the server'),
        (2, 'Login Failed', 'Security', 'Failed login attempt detected'),
        (3, 'Service Restarted', 'Maintenance', 'A service was restarted successfully')
    ]


    # Insert sample eventslog data with current timestamps
    event_logs = [
        (1, 2, datetime.datetime.now().isoformat(), 'Open'),
        (2, 3, datetime.datetime.now().isoformat(), 'Closed'),
        (3, 1, datetime.datetime.now().isoformat(), 'Open')
    ]

    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    c.executemany('INSERT OR IGNORE INTO severity VALUES (?, ?, ?)', severities)
    c.executemany('INSERT OR IGNORE INTO event VALUES (?, ?, ?, ?)', events)
    c.executemany('INSERT INTO eventslog VALUES (?, ?, ?, ?)', event_logs)

    conn.commit()
    conn.close()


##########################################################
def userquery_execute(db_name: str, userquery: str):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute(userquery)
    query_response = c.fetchall()
    conn.close()
    return query_response

##########################################################
def dump_table(db_name: str, table_name: str):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute(f"SELECT * FROM {table_name}")
    rows = c.fetchall()
    conn.close()
    return rows

##########################################################
def describe_table(db_name: str, table_name: str):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute(f'PRAGMA table_info({table_name})')
    columns = c.fetchall()
    print(f" directly from desc table: {columns}")
    conn.close()
    return columns

##########################################################
def describe_table_columns(table: str):
    #for table in ['severity', 'event', 'eventslog']:
    print(f"Schema for table '{table}':")
    schema = describe_table(DB_NAME, table)
    for col in schema:
        cid, name, col_type, notnull, default_value, pk = col
        print(f"  Column: {name}, Type: {col_type}, Not Null: {bool(notnull)}, "
            f"Default: {default_value}, Primary Key: {bool(pk)}")
    print()

##########################################################
# Usage example: create the tables in 'events_database.db'
if __name__ == "__main__":
    print(" ... STARTED ....  ")
    #create_tables(DB_NAME)
    print("---> Tables created successfully")

    print("Insert into table started ...")
    #insert_data_into_tables(DB_NAME)
    print("--->Inserted rows in dianostics.db/events, severity, eventslog tables completed")

    print(" DumpTable: severity STARTED ")
    # Dump severity data from severity table
    describe_table_columns('severity')
    severity_data = dump_table(DB_NAME, 'severity')
    print("Severity Table Data:")
    for row in severity_data:
        print(row)
    print(" DumpTable: severity END ")

    print(" DumpTable: event STARTED ")
    describe_table_columns('event')
    # Dump data from event table 
    event_data = dump_table(DB_NAME, 'event')
    print("\nEvent Table Data:")
    for row in event_data:
        print(row)
    print(" DumpTable: event END ")

    print(" DumpTable: eventslog STARTED ")
    describe_table_columns('eventslog')
    # Dump data from eventslog table
    eventslog_data = dump_table(DB_NAME, 'eventslog')
    print("\nEventslog Table Data:")
    for row in eventslog_data:
        print(row)
    print(" DumpTable: eventslog END ")

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
    userquery_results = userquery_execute(DB_NAME, query)
    for row in userquery_results:
        print(row)
    print(" DumpUserQuery RESULTS  END ")

    print(" ... END ....  ")

