# net start mysql80 #启动MySQL
# mysql -u root -p密码 #输入密码进行登录
# quit #退出

create database 名称 default charset utf8mb4 ;#创建一个数据库
drop database 名称 ;#删除某个数据库

show databases ;#展示所有数据库
use test ;#指定使用某个数据库

show tables ;#查看当前表的数据结构

CREATE TABLE 表名(
	id int COMMENT ' id ' ,
	name varchar(10) COMMENT ' 姓名' ,
	age int COMMENT ' 年龄 '
) COMMENT ' 表注释 ' ;			#创建一个列表

desc table1; #查询某个表的结构
show create table table1;#查看表的详细信息

insert into table1(id, name, age) values (1,'yyl',22) ; #为表中的多列添加字段

insert into table1 values (2,'asd',18);#也可以不指定名称，按顺序填写即可

insert into table1 values (2,'sald',13),(4,'asas',28),(5,'ad',43);#添加多行字段

update table1 set name='yyl' where id=5;#将id为5的字段里面的name修改为yyl,如果不加后面的where字段就会更新整个表的值

delete from table1 where id=4;#删除id为4的行

select id,name from table1 ; #查询某几个名称的字段

select name as '姓名' from table1 ; #可为查询的字段起别名

select distinct name as '姓名' from table1 ;#不会重复显示

select * from table1 where name='yyl';#查询姓名等于yyl的字段

select * from table1 where name like '___';#查询姓名为三个字的字段

select * from table1 where name like '%l';#查询name最后一个字段为l的字段

select count(name) from table1;#统计name的个数
select avg(age) from table1;#统计age的平均

select id ,count(*) from table1 group by id;#查询每一个类型的id的数量
#3. 查询年龄小于20的员工,并根据姓名分组 ,并找出出现次数大于1的姓名，并显示数量
select name, count(*) from table1 where age < 20 group by name having count(*) >= 1;

select * from table1 order by id asc;#按照id大小进行升序排序（desc降序）
select * from table1 order by id asc , age desc ;#先按照id升序，如果id相同再按照age降序

select * from table1 limit 0,2;#在第一页显示前两条数据
select * from table1 limit 2,3;#在第2页显示3条数据,从第二条字段开始