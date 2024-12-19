# import datetime
# import sqlite3

# # 记录收支情况的类


# class FinanceRecord:
#     def __init__(self, amount, category, date=None):
#         self.amount = amount
#         self.category = category
#         self.date = date if date else datetime.datetime.now().strftime(
#             "%Y-%m-%d %H:%M:%S")  # 转换为字符串

#     def __str__(self):
#         return f"{self.date}: {self.category} - {self.amount}"


# class LifeEvent:
#     def __init__(self, description, date=None):
#         self.description = description
#         self.date = date if date else datetime.datetime.now().strftime(
#             "%Y-%m-%d %H:%M:%S")  # 转换为字符串

#     def __str__(self):
#         return f"{self.date}: {self.description}"


# class HealthRecord:
#     def __init__(self, health_data, date=None):
#         self.health_data = health_data
#         self.date = date if date else datetime.datetime.now().strftime(
#             "%Y-%m-%d %H:%M:%S")  # 转换为字符串

#     def __str__(self):
#         return f"{self.date}: {self.health_data}"


# class ContactRecord:
#     def __init__(self, name, phone, email, date=None):
#         self.name = name
#         self.phone = phone
#         self.email = email
#         self.date = date if date else datetime.datetime.now().strftime(
#             "%Y-%m-%d %H:%M:%S")  # 转换为字符串

#     def __str__(self):
#         return f"{self.date}: {self.name} - {self.phone} - {self.email}"


# class PersonalAssistant:
#     def __init__(self):
#         self.finance_records = []
#         self.life_events = []
#         self.health_records = []
#         self.contact_records = []
#         self.conn = sqlite3.connect(':memory:')  # 使用内存数据库
#         self.create_tables()

#     def create_tables(self):
#         cursor = self.conn.cursor()
#         cursor.execute('''CREATE TABLE finance_records
#                           (amount REAL, category TEXT, date TEXT)''')
#         cursor.execute('''CREATE TABLE life_events
#                           (description TEXT, date TEXT)''')
#         cursor.execute('''CREATE TABLE health_records
#                           (health_data TEXT, date TEXT)''')
#         cursor.execute('''CREATE TABLE contact_records
#                           (name TEXT, phone TEXT, email TEXT, date TEXT)''')
#         self.conn.commit()

#     def add_finance_record(self, amount, category):
#         record = FinanceRecord(amount, category)
#         self.finance_records.append(record)
#         cursor = self.conn.cursor()
#         cursor.execute("INSERT INTO finance_records (amount, category, date) VALUES (?, ?, ?)",
#                        (record.amount, record.category, record.date))  # 存储为字符串
#         self.conn.commit()

#     def delete_finance_record(self, amount, category):
#         self.finance_records = [record for record in self.finance_records if not (
#             record.amount == amount and record.category == category)]
#         cursor = self.conn.cursor()
#         cursor.execute(
#             "DELETE FROM finance_records WHERE amount=? AND category=?", (amount, category))
#         self.conn.commit()

#     def update_finance_record(self, old_amount, old_category, new_amount, new_category):
#         for record in self.finance_records:
#             if record.amount == old_amount and record.category == old_category:
#                 record.amount = new_amount
#                 record.category = new_category
#         cursor = self.conn.cursor()
#         cursor.execute("UPDATE finance_records SET amount=?, category=? WHERE amount=? AND category=?",
#                        (new_amount, new_category, old_amount, old_category))
#         self.conn.commit()

#     def get_finance_records(self):
#         return self.finance_records

#     def get_finance_records_from_db(self):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT * FROM finance_records")
#         return cursor.fetchall()

#     def add_life_event(self, description):
#         event = LifeEvent(description)
#         self.life_events.append(event)
#         cursor = self.conn.cursor()
#         cursor.execute("INSERT INTO life_events (description, date) VALUES (?, ?)",
#                        (event.description, event.date))  # 存储为字符串
#         self.conn.commit()

#     def delete_life_event(self, description):
#         self.life_events = [
#             event for event in self.life_events if event.description != description]
#         cursor = self.conn.cursor()
#         cursor.execute(
#             "DELETE FROM life_events WHERE description=?", (description,))
#         self.conn.commit()

#     def update_life_event(self, old_description, new_description):
#         for event in self.life_events:
#             if event.description == old_description:
#                 event.description = new_description
#         cursor = self.conn.cursor()
#         cursor.execute("UPDATE life_events SET description=? WHERE description=?",
#                        (new_description, old_description))
#         self.conn.commit()

#     def get_life_events(self):
#         return self.life_events

#     def get_life_events_from_db(self):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT * FROM life_events")
#         return cursor.fetchall()

#     def add_health_record(self, health_data):
#         record = HealthRecord(health_data)
#         self.health_records.append(record)
#         cursor = self.conn.cursor()
#         cursor.execute("INSERT INTO health_records (health_data, date) VALUES (?, ?)",
#                        (record.health_data, record.date))  # 存储为字符串
#         self.conn.commit()

#     def delete_health_record(self, health_data):
#         self.health_records = [
#             record for record in self.health_records if record.health_data != health_data]
#         cursor = self.conn.cursor()
#         cursor.execute(
#             "DELETE FROM health_records WHERE health_data=?", (health_data,))
#         self.conn.commit()

#     def update_health_record(self, old_health_data, new_health_data):
#         for record in self.health_records:
#             if record.health_data == old_health_data:
#                 record.health_data = new_health_data
#         cursor = self.conn.cursor()
#         cursor.execute("UPDATE health_records SET health_data=? WHERE health_data=?",
#                        (new_health_data, old_health_data))
#         self.conn.commit()

#     def get_health_records(self):
#         return self.health_records

#     def get_health_records_from_db(self):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT * FROM health_records")
#         return cursor.fetchall()

#     def add_contact_record(self, name, phone, email):
#         record = ContactRecord(name, phone, email)
#         self.contact_records.append(record)
#         cursor = self.conn.cursor()
#         cursor.execute("INSERT INTO contact_records (name, phone, email, date) VALUES (?, ?, ?, ?)",
#                        (record.name, record.phone, record.email, record.date))  # 存储为字符串
#         self.conn.commit()

#     def delete_contact_record(self, name):
#         self.contact_records = [
#             record for record in self.contact_records if record.name != name]
#         cursor = self.conn.cursor()
#         cursor.execute("DELETE FROM contact_records WHERE name=?", (name,))
#         self.conn.commit()

#     def update_contact_record(self, old_name, new_name, new_phone, new_email):
#         for contact in self.contact_records:
#             if contact.name == old_name:
#                 contact.name = new_name
#                 contact.phone = new_phone
#                 contact.email = new_email
#         cursor = self.conn.cursor()
#         cursor.execute("UPDATE contact_records SET name=?, phone=?, email=? WHERE name=?",
#                        (new_name, new_phone, new_email, old_name))
#         self.conn.commit()

#     def get_contact_records(self):
#         return self.contact_records

#     def get_contact_records_from_db(self):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT * FROM contact_records")
#         return cursor.fetchall()


# # 第二版
# import datetime
# import sqlite3

# # 记录收支情况的类


# class FinanceRecord:
#     def __init__(self, amount, category, date=None):
#         self.amount = amount
#         self.category = category
#         self.date = date if date else datetime.datetime.now()

#     def __str__(self):
#         return f"{self.date}: {self.category} - {self.amount}"


# class LifeEvent:
#     def __init__(self, description, date=None):
#         self.description = description
#         self.date = date if date else datetime.datetime.now()

#     def __str__(self):
#         return f"{self.date}: {self.description}"


# class HealthRecord:
#     def __init__(self, health_data, date=None):
#         self.health_data = health_data
#         self.date = date if date else datetime.datetime.now()

#     def __str__(self):
#         return f"{self.date}: {self.health_data}"


# class ContactRecord:
#     def __init__(self, name, phone, email, date=None):
#         self.name = name
#         self.phone = phone
#         self.email = email
#         self.date = date if date else datetime.datetime.now()

#     def __str__(self):
#         return f"{self.date}: {self.name} - {self.phone} - {self.email}"


# class PersonalAssistant:
#     def __init__(self):
#         self.finance_records = []
#         self.life_events = []
#         self.health_records = []
#         self.contact_records = []
#         self.conn = sqlite3.connect(':memory:')  # 使用内存数据库
#         self.create_tables()

#     def create_tables(self):
#         cursor = self.conn.cursor()
#         cursor.execute('''CREATE TABLE finance_records
#                           (amount REAL, category TEXT, date TEXT)''')
#         cursor.execute('''CREATE TABLE life_events
#                           (description TEXT, date TEXT)''')
#         cursor.execute('''CREATE TABLE health_records
#                           (health_data TEXT, date TEXT)''')
#         cursor.execute('''CREATE TABLE contact_records
#                           (name TEXT, phone TEXT, email TEXT, date TEXT)''')
#         self.conn.commit()

#     def add_finance_record(self, amount, category):
#         record = FinanceRecord(amount, category)
#         self.finance_records.append(record)
#         cursor = self.conn.cursor()
#         cursor.execute("INSERT INTO finance_records (amount, category, date) VALUES (?, ?, ?)",
#                        (record.amount, record.category, record.date.strftime("%Y.%m.%d %H:%M:%S")))
#         self.conn.commit()

#     def delete_finance_record(self, amount, category):
#         self.finance_records = [record for record in self.finance_records if not (
#             record.amount == amount and record.category == category)]
#         cursor = self.conn.cursor()
#         cursor.execute(
#             "DELETE FROM finance_records WHERE amount=? AND category=?", (amount, category))
#         self.conn.commit()

#     def update_finance_record(self, old_amount, old_category, new_amount, new_category):
#         for record in self.finance_records:
#             if record.amount == old_amount and record.category == old_category:
#                 record.amount = new_amount
#                 record.category = new_category
#         cursor = self.conn.cursor()
#         cursor.execute("UPDATE finance_records SET amount=?, category=? WHERE amount=? AND category=?",
#                        (new_amount, new_category, old_amount, old_category))
#         self.conn.commit()

#     def get_finance_records(self):
#         return self.finance_records

#     def get_finance_records_from_db(self):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT * FROM finance_records")
#         return cursor.fetchall()

#     def add_life_event(self, description):
#         event = LifeEvent(description)
#         self.life_events.append(event)
#         cursor = self.conn.cursor()
#         cursor.execute("INSERT INTO life_events (description, date) VALUES (?, ?)",
#                        (event.description, event.date.strftime("%Y.%m.%d %H:%M:%S")))
#         self.conn.commit()

#     def delete_life_event(self, description):
#         self.life_events = [
#             event for event in self.life_events if event.description != description]
#         cursor = self.conn.cursor()
#         cursor.execute(
#             "DELETE FROM life_events WHERE description=?", (description,))
#         self.conn.commit()

#     def update_life_event(self, old_description, new_description):
#         for event in self.life_events:
#             if event.description == old_description:
#                 event.description = new_description
#         cursor = self.conn.cursor()
#         cursor.execute("UPDATE life_events SET description=? WHERE description=?",
#                        (new_description, old_description))
#         self.conn.commit()

#     def get_life_events(self):
#         return self.life_events

#     def get_life_events_from_db(self):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT * FROM life_events")
#         return cursor.fetchall()

#     def add_health_record(self, health_data):
#         record = HealthRecord(health_data)
#         self.health_records.append(record)
#         cursor = self.conn.cursor()
#         cursor.execute("INSERT INTO health_records (health_data, date) VALUES (?, ?)",
#                        (record.health_data, record.date.strftime("%Y.%m.%d %H:%M:%S")))
#         self.conn.commit()

#     def delete_health_record(self, health_data):
#         self.health_records = [
#             record for record in self.health_records if record.health_data != health_data]
#         cursor = self.conn.cursor()
#         cursor.execute(
#             "DELETE FROM health_records WHERE health_data=?", (health_data,))
#         self.conn.commit()

#     def update_health_record(self, old_health_data, new_health_data):
#         for record in self.health_records:
#             if record.health_data == old_health_data:
#                 record.health_data = new_health_data
#         cursor = self.conn.cursor()
#         cursor.execute("UPDATE health_records SET health_data=? WHERE health_data=?",
#                        (new_health_data, old_health_data))
#         self.conn.commit()

#     def get_health_records(self):
#         return self.health_records

#     def get_health_records_from_db(self):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT * FROM health_records")
#         return cursor.fetchall()

#     def add_contact_record(self, name, phone, email):
#         record = ContactRecord(name, phone, email)
#         self.contact_records.append(record)
#         cursor = self.conn.cursor()
#         cursor.execute("INSERT INTO contact_records (name, phone, email, date) VALUES (?, ?, ?, ?)",
#                        (record.name, record.phone, record.email, record.date.strftime("%Y.%m.%d %H:%M:%S")))
#         self.conn.commit()

#     def delete_contact_record(self, name):
#         self.contact_records = [
#             record for record in self.contact_records if record.name != name]
#         cursor = self.conn.cursor()
#         cursor.execute("DELETE FROM contact_records WHERE name=?", (name,))
#         self.conn.commit()

#     def update_contact_record(self, old_name, new_name, new_phone, new_email):
#         for contact in self.contact_records:
#             if contact.name == old_name:
#                 contact.name = new_name
#                 contact.phone = new_phone
#                 contact.email = new_email
#         cursor = self.conn.cursor()
#         cursor.execute("UPDATE contact_records SET name=?, phone=?, email=? WHERE name=?",
#                        (new_name, new_phone, new_email, old_name))
#         self.conn.commit()

#     def get_contact_records(self):
#         return self.contact_records

#     def get_contact_records_from_db(self):
#         cursor = self.conn.cursor()
#         cursor.execute("SELECT * FROM contact_records")
#         return cursor.fetchall()


# 第三版
import datetime
import sqlite3

# 记录收支情况的类


class FinanceRecord:
    def __init__(self, amount, category, date=None):
        self.amount = amount
        self.category = category
        self.date = date if date else datetime.datetime.now()

    def __str__(self):
        return f"{self.date}: {self.category} - {self.amount}"


class LifeEvent:
    def __init__(self, description, date=None):
        self.description = description
        self.date = date if date else datetime.datetime.now()

    def __str__(self):
        return f"{self.date}: {self.description}"


class HealthRecord:
    def __init__(self, health_data, date=None):
        self.health_data = health_data
        self.date = date if date else datetime.datetime.now()

    def __str__(self):
        return f"{self.date}: {self.health_data}"


class ContactRecord:
    def __init__(self, name, phone, email, date=None):
        self.name = name
        self.phone = phone
        self.email = email
        self.date = date if date else datetime.datetime.now()

    def __str__(self):
        return f"{self.date}: {self.name} - {self.phone} - {self.email}"


class PersonalAssistant:
    def __init__(self):
        self.finance_records = []
        self.life_events = []
        self.health_records = []
        self.contact_records = []
        self.conn = sqlite3.connect(':memory:')  # 使用内存数据库
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE finance_records
                          (amount REAL, category TEXT, date TEXT)''')
        cursor.execute('''CREATE TABLE life_events
                          (description TEXT, date TEXT)''')
        cursor.execute('''CREATE TABLE health_records
                          (health_data TEXT, date TEXT)''')
        cursor.execute('''CREATE TABLE contact_records
                          (name TEXT, phone TEXT, email TEXT, date TEXT)''')
        self.conn.commit()

    def add_finance_record(self, amount, category):
        record = FinanceRecord(amount, category)
        self.finance_records.append(record)
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO finance_records (amount, category, date) VALUES (?, ?, ?)",
                       (record.amount, record.category, record.date.strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()

    def delete_finance_record(self, amount, category):
        self.finance_records = [record for record in self.finance_records if not (
            record.amount == amount and record.category == category)]
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM finance_records WHERE amount=? AND category=?", (amount, category))
        self.conn.commit()

    def update_finance_record(self, old_amount, old_category, new_amount, new_category):
        for record in self.finance_records:
            if record.amount == old_amount and record.category == old_category:
                record.amount = new_amount
                record.category = new_category
        cursor = self.conn.cursor()
        cursor.execute("UPDATE finance_records SET amount=?, category=? WHERE amount=? AND category=?",
                       (new_amount, new_category, old_amount, old_category))
        self.conn.commit()

    def get_finance_records(self):
        return self.finance_records

    def get_finance_records_from_db(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM finance_records")
        return cursor.fetchall()

    def add_life_event(self, description):
        event = LifeEvent(description)
        self.life_events.append(event)
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO life_events (description, date) VALUES (?, ?)",
                       (event.description, event.date.strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()

    def delete_life_event(self, description):
        self.life_events = [
            event for event in self.life_events if event.description != description]
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM life_events WHERE description=?", (description,))
        self.conn.commit()

    def update_life_event(self, old_description, new_description):
        for event in self.life_events:
            if event.description == old_description:
                event.description = new_description
        cursor = self.conn.cursor()
        cursor.execute("UPDATE life_events SET description=? WHERE description=?",
                       (new_description, old_description))
        self.conn.commit()

    def get_life_events(self):
        return self.life_events

    def get_life_events_from_db(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM life_events")
        return cursor.fetchall()

    def add_health_record(self, health_data):
        record = HealthRecord(health_data)
        self.health_records.append(record)
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO health_records (health_data, date) VALUES (?, ?)",
                       (record.health_data, record.date.strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()

    def delete_health_record(self, health_data):
        self.health_records = [
            record for record in self.health_records if record.health_data != health_data]
        cursor = self.conn.cursor()
        cursor.execute(
            "DELETE FROM health_records WHERE health_data=?", (health_data,))
        self.conn.commit()

    def update_health_record(self, old_health_data, new_health_data):
        for record in self.health_records:
            if record.health_data == old_health_data:
                record.health_data = new_health_data
        cursor = self.conn.cursor()
        cursor.execute("UPDATE health_records SET health_data=? WHERE health_data=?",
                       (new_health_data, old_health_data))
        self.conn.commit()

    def get_health_records(self):
        return self.health_records

    def get_health_records_from_db(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM health_records")
        return cursor.fetchall()

    def add_contact_record(self, name, phone, email):
        record = ContactRecord(name, phone, email)
        self.contact_records.append(record)
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO contact_records (name, phone, email, date) VALUES (?, ?, ?, ?)",
                       (record.name, record.phone, record.email, record.date.strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()

    def delete_contact_record(self, name):
        self.contact_records = [
            record for record in self.contact_records if record.name != name]
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM contact_records WHERE name=?", (name,))
        self.conn.commit()

    def update_contact_record(self, old_name, new_name, new_phone, new_email):
        for contact in self.contact_records:
            if contact.name == old_name:
                contact.name = new_name
                contact.phone = new_phone
                contact.email = new_email
        cursor = self.conn.cursor()
        cursor.execute("UPDATE contact_records SET name=?, phone=?, email=? WHERE name=?",
                       (new_name, new_phone, new_email, old_name))
        self.conn.commit()

    def get_contact_records(self):
        return self.contact_records

    def get_contact_records_from_db(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM contact_records")
        return cursor.fetchall()
