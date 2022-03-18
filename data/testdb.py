import pymysql

db = pymysql.connect(host='127.0.0.1', port=3306, user='root', db='arrl', charset='utf8')

cursor = db.cursor()

cursor.execute("select version()")
data = cursor.fetchone()
print("Database Version: %s"%data)

print("Create database test")
cursor.execute("drop database if exists test")
sql = "create database test"
cursor.execute(sql)

print("Create table employee")
cursor.execute("drop table if exists employee")
sql = """
create table employee(
first_name char(20) not null,
last_name char(20),
age int,
sex char(1),
income float)
"""
cursor.execute(sql)
sql = "select * from employee"
cursor.execute(sql)
data = cursor.fetchone()
print(data)

print("Insert into table")
sql = "insert into employee values('Li','Mei',20,'W',5000)"
cursor.execute(sql)
db.commit()
sql = "select * from employee"
cursor.execute(sql)
data = cursor.fetchone()
print(data)

print("Query from table")
sql = "select * from employee where income > '%d'"%(1000)
cursor.execute(sql)
data = cursor.fetchone()
print(data)

print("Update table")
sql = "update employee set age = age+1 where sex = '%c'"%('W')
cursor.execute(sql)
db.commit()
sql = "select * from employee"
cursor.execute(sql)
data = cursor.fetchone()
print(data)

print("Delete from table")
sql = "delete from employee where age > '%d'"%(30)
cursor.execute(sql)
db.commit()
sql = "select * from employee"
cursor.execute(sql)
data = cursor.fetchone()
print(data)

print("Close")
db.close()


