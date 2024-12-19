import pytest
from personal_assistant import PersonalAssistant

# 财务记录的测试用例


def test_add_finance_record():
    pa = PersonalAssistant()
    pa.add_finance_record(100, "Groceries")
    records = pa.get_finance_records()
    assert len(records) == 1
    assert records[0].amount == 100
    assert records[0].category == "Groceries"
    db_records = pa.get_finance_records_from_db()
    assert len(db_records) == 1
    assert db_records[0][0] == 100
    assert db_records[0][1] == "Groceries"
    assert db_records[0][2] == records[0].date.strftime("%Y-%m-%d %H:%M:%S")


def test_delete_finance_record():
    pa = PersonalAssistant()
    pa.add_finance_record(100, "Groceries")
    pa.delete_finance_record(100, "Groceries")
    records = pa.get_finance_records()
    assert len(records) == 0
    db_records = pa.get_finance_records_from_db()
    assert len(db_records) == 0


def test_update_finance_record():
    pa = PersonalAssistant()
    pa.add_finance_record(100, "Groceries")
    pa.update_finance_record(100, "Groceries", 150, "Utilities")
    records = pa.get_finance_records()
    assert len(records) == 1
    assert records[0].amount == 150
    assert records[0].category == "Utilities"
    db_records = pa.get_finance_records_from_db()
    assert len(db_records) == 1
    assert db_records[0][0] == 150
    assert db_records[0][1] == "Utilities"


def test_get_finance_records():
    pa = PersonalAssistant()
    pa.add_finance_record(100, "Groceries")
    pa.add_finance_record(200, "Utilities")
    records = pa.get_finance_records()
    assert len(records) == 2
    assert records[0].amount == 100
    assert records[0].category == "Groceries"
    assert records[1].amount == 200
    assert records[1].category == "Utilities"
    db_records = pa.get_finance_records_from_db()
    assert len(db_records) == 2
    assert db_records[0][0] == 100
    assert db_records[0][1] == "Groceries"
    assert db_records[1][0] == 200
    assert db_records[1][1] == "Utilities"

# 生活事件的测试用例


def test_add_life_event():
    pa = PersonalAssistant()
    pa.add_life_event("Went to the gym")
    events = pa.get_life_events()
    assert len(events) == 1
    assert events[0].description == "Went to the gym"
    db_events = pa.get_life_events_from_db()
    assert len(db_events) == 1
    assert db_events[0][0] == "Went to the gym"
    assert db_events[0][1] == events[0].date.strftime("%Y-%m-%d %H:%M:%S")


def test_delete_life_event():
    pa = PersonalAssistant()
    pa.add_life_event("Went to the gym")
    pa.delete_life_event("Went to the gym")
    events = pa.get_life_events()
    assert len(events) == 0
    db_events = pa.get_life_events_from_db()
    assert len(db_events) == 0


def test_update_life_event():
    pa = PersonalAssistant()
    pa.add_life_event("Went to the gym")
    pa.update_life_event("Went to the gym", "Went to the park")
    events = pa.get_life_events()
    assert len(events) == 1
    assert events[0].description == "Went to the park"
    db_events = pa.get_life_events_from_db()
    assert len(db_events) == 1
    assert db_events[0][0] == "Went to the park"


def test_get_life_events():
    pa = PersonalAssistant()
    pa.add_life_event("Went to the gym")
    pa.add_life_event("Had dinner with friends")
    events = pa.get_life_events()
    assert len(events) == 2
    assert events[0].description == "Went to the gym"
    assert events[1].description == "Had dinner with friends"
    db_events = pa.get_life_events_from_db()
    assert len(db_events) == 2
    assert db_events[0][0] == "Went to the gym"
    assert db_events[1][0] == "Had dinner with friends"

# 健康记录的测试用例


def test_add_health_record():
    pa = PersonalAssistant()
    pa.add_health_record("Blood pressure normal")
    records = pa.get_health_records()
    assert len(records) == 1
    assert records[0].health_data == "Blood pressure normal"
    db_records = pa.get_health_records_from_db()
    assert len(db_records) == 1
    assert db_records[0][0] == "Blood pressure normal"
    assert db_records[0][1] == records[0].date.strftime("%Y-%m-%d %H:%M:%S")


def test_delete_health_record():
    pa = PersonalAssistant()
    pa.add_health_record("Blood pressure normal")
    pa.delete_health_record("Blood pressure normal")
    records = pa.get_health_records()
    assert len(records) == 0
    db_records = pa.get_health_records_from_db()
    assert len(db_records) == 0


def test_update_health_record():
    pa = PersonalAssistant()
    pa.add_health_record("Blood pressure normal")
    pa.update_health_record("Blood pressure normal", "Blood pressure high")
    records = pa.get_health_records()
    assert len(records) == 1
    assert records[0].health_data == "Blood pressure high"
    db_records = pa.get_health_records_from_db()
    assert len(db_records) == 1
    assert db_records[0][0] == "Blood pressure high"


def test_get_health_records():
    pa = PersonalAssistant()
    pa.add_health_record("Blood pressure normal")
    pa.add_health_record("Cholesterol level high")
    records = pa.get_health_records()
    assert len(records) == 2
    assert records[0].health_data == "Blood pressure normal"
    assert records[1].health_data == "Cholesterol level high"
    db_records = pa.get_health_records_from_db()
    assert len(db_records) == 2
    assert db_records[0][0] == "Blood pressure normal"
    assert db_records[1][0] == "Cholesterol level high"

# 联系人记录的测试用例


def test_add_contact_record():
    pa = PersonalAssistant()
    pa.add_contact_record("John Doe", "1234567890", "john.doe@example.com")
    records = pa.get_contact_records()
    assert len(records) == 1
    assert records[0].name == "John Doe"
    assert records[0].phone == "1234567890"
    assert records[0].email == "john.doe@example.com"
    db_records = pa.get_contact_records_from_db()
    assert len(db_records) == 1
    assert db_records[0][0] == "John Doe"
    assert db_records[0][1] == "1234567890"
    assert db_records[0][2] == "john.doe@example.com"
    assert db_records[0][3] == records[0].date.strftime("%Y-%m-%d %H:%M:%S")


def test_delete_contact_record():
    pa = PersonalAssistant()
    pa.add_contact_record("John Doe", "1234567890", "john.doe@example.com")
    pa.delete_contact_record("John Doe")
    records = pa.get_contact_records()
    assert len(records) == 0
    db_records = pa.get_contact_records_from_db()
    assert len(db_records) == 0


def test_update_contact_record():
    pa = PersonalAssistant()
    pa.add_contact_record("John Doe", "1234567890", "john.doe@example.com")
    pa.update_contact_record("John Doe", "Jane Doe",
                             "0987654321", "jane.doe@example.com")
    records = pa.get_contact_records()
    assert len(records) == 1
    assert records[0].name == "Jane Doe"
    assert records[0].phone == "0987654321"
    assert records[0].email == "jane.doe@example.com"
    db_records = pa.get_contact_records_from_db()
    assert len(db_records) == 1
    assert db_records[0][0] == "Jane Doe"
    assert db_records[0][1] == "0987654321"
    assert db_records[0][2] == "jane.doe@example.com"


def test_get_contact_records():
    pa = PersonalAssistant()
    pa.add_contact_record("John Doe", "1234567890", "john.doe@example.com")
    pa.add_contact_record("Jane Doe", "0987654321", "jane.doe@example.com")
    records = pa.get_contact_records()
    assert len(records) == 2
    assert records[0].name == "John Doe"
    assert records[0].phone == "1234567890"
    assert records[0].email == "john.doe@example.com"
    assert records[1].name == "Jane Doe"
    assert records[1].phone == "0987654321"
    assert records[1].email == "jane.doe@example.com"
    db_records = pa.get_contact_records_from_db()
    assert len(db_records) == 2
    assert db_records[0][0] == "John Doe"
    assert db_records[0][1] == "1234567890"
    assert db_records[0][2] == "john.doe@example.com"
    assert db_records[0][3] == records[0].date.strftime("%Y-%m-%d %H:%M:%S")
    assert db_records[1][0] == "Jane Doe"
    assert db_records[1][1] == "0987654321"
    assert db_records[1][2] == "jane.doe@example.com"
    assert db_records[1][3] == records[1].date.strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    pytest.main([__file__])
