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
        (1, 'Preventative', 'Preventative measures or minor localized faults'),
        (2, 'Warning', 'Formal alerts, system resources are fully committed and reliability is threatened'),
        (3, 'Emergency', 'Formal emergency declaration, load loss is probable or actively occurring')
    ]


    # Insert sample event data
    events = [
        (1, 'Disk Space Low', 'System', 'Disk space is running low on the server'),
        (2, 'Login Failed', 'Security', 'Failed login attempt detected'),
        (3, 'Service Restarted', 'Maintenance', 'A service was restarted successfully')
        (1, 'Flex Alert Issued', 'Grid Operation (Preventative)', 'A public call for voluntary energy conservation to reduce demand'),
        (2, 'Restricted Maintenance Operations (RMO)', 'Grid Operation (Preventative)', 'Order requiring operators to postpone planned outages to ensure asset availability'),
        (3, 'EEA Watch Issued', 'Grid Emergency (Energy)', 'Day-ahead or real-time analysis shows energy deficiencies are expected'),
        (4, 'EEA 1 Issued', 'Grid Emergency (Energy)', 'Real-time analysis shows all resources are in use and deficiencies are expected'),
        (5, 'EEA 2 Issued', 'Grid Emergency (Energy)', 'ISO requests emergency energy and has activated emergency demand response programs'),
        (6, 'EEA 3 (Preparation for Outages)', 'Grid Emergency (Energy)', 'ISO unable to meet minimum reliability reserves; utilities alerted to prepare for outages'),
        (7, 'EEA 3 (Ordering Rotating Outages)', 'Grid Emergency (Energy)', 'ISO has ordered utilities to begin rotating power outages (firm load shed)'),
        (8, 'Transmission Emergency Declared', 'Grid Emergency (Transmission)', 'Declared for any event threatening or limiting transmission grid capability'),
        (9, 'Forced Generation Outage', 'Asset Outage (Generation)', 'An unplanned, sudden generator outage due to equipment failure or other notice'),
        (10, 'Planned Generation Outage', 'Asset Outage (Generation)', 'A generator outage submitted at least seven days in advance for maintenance'),
        (11, 'Forced Line Outage (Vegetation)', 'Asset Outage (Transmission)', 'A transmission line is forced offline due to contact with vegetation'),
        (12, 'Forced Line Outage (Substation Equipment)', 'Asset Outage (Transmission)', 'A line outage caused by a failure of substation equipment'),
        (13, 'Forced Line Outage (Circuit Breaker)', 'Asset Outage (Transmission)', 'A line outage specifically attributed to "Circuit Breaker Trouble"'),
        (14, 'Forced Line Outage (Protection)', 'Asset Outage (Transmission)', 'An outage caused by the operation of a protection system (e.g., a relay)'),
        (15, 'Forced Line Outage (Other/Weather)', 'Asset Outage (Transmission)', 'An outage caused by other factors, such as high winds, wildfires, or vandalism'),
        (16, 'Emergency Demand Response Dispatch', 'Grid Operation (Preventative)', 'Activation of formal, out-of-market demand response programs')
    ]


    # Insert sample eventslog data with current timestamps
    event_logs = [
        ( 1 ,  5 ,  3 ,  '2021-07-09T21:30:00' ,  'ENDED' ),
        ( 2 ,  4 ,  2 ,  '2021-07-09T22:00:00' ,  'ENDED' ),
        ( 3 ,  1 ,  1 ,  '2022-09-05T15:00:00' ,  'DECLARED' ),
        ( 4 ,  3 ,  2 ,  '2022-09-05T17:00:00' ,  'DECLARED' ), 
        ( 5 ,  4 ,  2 ,  '2022-09-05T17:00:00' ,  'DECLARED' ),
        ( 6 ,  1 ,  1 ,  '2022-09-06T15:00:00' ,  'DECLARED' ),
        ( 7 ,  5 ,  3 ,  '2022-09-06T16:00:00' ,  'DECLARED' ),
        ( 8 ,  6 ,  3 ,  '2022-09-06T17:17:00' ,  'DECLARED' ), 
        ( 9 ,  16 ,  3 ,  '2022-09-06T17:45:00' ,  'DECLARED' ),
        ( 10 ,  6 ,  3 ,  '2022-09-06T21:00:00' ,  'ENDED' ),
        ( 11 ,  4 ,  2 ,  '2023-07-20T19:30:00' ,  'DECLARED' ),
        ( 12 ,  4 ,  2 ,  '2023-07-20T22:00:00' ,  'ENDED' ),
        ( 13 ,  3 ,  2 ,  '2023-07-25T19:26:00' ,  'DECLARED' ),
        ( 14 ,  3 ,  2 ,  '2023-07-25T23:59:00' ,  'ENDED' ),
        ( 15 ,  2 ,  1 ,  '2024-01-18T06:00:00' ,  'DECLARED' ),
        ( 16 ,  2 ,  1 ,  '2024-01-21T23:59:00' ,  'ENDED' ),
        ( 17 ,  8 ,  3 ,  '2024-03-01T08:50:00' ,  'DECLARED' ),
        ( 18 ,  8 ,  3 ,  '2024-03-04T20:44:00' ,  'ENDED' ),
        ( 19 ,  2 ,  1 ,  '2024-07-03T00:01:00' ,  'DECLARED' ),
        ( 20 ,  2 ,  1 ,  '2024-07-07T23:59:00' ,  'ENDED' ),
        ( 21 ,  8 ,  3 ,  '2024-07-18T14:35:00' ,  'DECLARED' ),
        ( 22 ,  8 ,  3 ,  '2024-07-18T23:59:00' ,  'ENDED' ),
        ( 23 ,  11 ,  1 ,  '2024-05-10T14:30:00' ,  'DECLARED' ),
        ( 24 ,  11 ,  1 ,  '2024-05-10T18:00:00' ,  'ENDED' ),
        ( 25 ,  12 ,  2 ,  '2024-06-15T09:15:00' ,  'DECLARED' ),
        ( 26 ,  12 ,  2 ,  '2024-06-16T03:00:00' ,  'ENDED' ),
        ( 27 ,  13 ,  1 ,  '2024-08-01T11:05:00' ,  'DECLARED' ),
        ( 28 ,  13 ,  1 ,  '2024-08-01T16:20:00' ,  'ENDED' ),
        ( 29 ,  15 ,  2 ,  '2025-01-20T08:00:00' ,  'DECLARED' ),
        ( 30 ,  15 ,  2 ,  '2025-01-20T17:00:00' ,  'ENDED' ),
        ( 31 ,  10 ,  1 ,  '2025-11-02T07:00:00' ,  'DECLARED' ),
        ( 32 ,  10 ,  1 ,  '2025-11-05T17:00:00' ,  'ENDED' ), 
        ( 33 ,  9 ,  2 ,  '2025-11-02T10:15:00' ,  'DECLARED' ),
        ( 34 ,  9 ,  2 ,  '2025-11-03T16:00:00' ,  'ENDED' ),
        ( 35 ,  9 ,  1 ,  '2025-11-02T11:00:00' ,  'DECLARED' ),
        ( 36 ,  9 ,  1 ,  '2025-11-03T09:00:00' ,  'DECLARED' )
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

