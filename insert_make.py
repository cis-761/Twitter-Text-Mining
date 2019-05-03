import pandas as pd


def tweets_table_insert():
    # first I am going to work on making the inserts from the CSV for Tweets Table
    '''
    id SERIAL PRIMARY KEY NOT NULL,
	text VARCHAR(280) NOT NULL,
	location VARCHAR,
	favorite BOOLEAN NOT NULL,
    date TIMESTAMP NOT NULL,
    rt BOOLEAN NOT NULL,
    '''

    #syntax for insert
    '''
    INSERT INTO TABLE_NAME (column1, column2, column3,...columnN)
    VALUES (value1, value2, value3,...valueN);
    '''
    data = pd.read_csv('./cleaned_data/tweets.csv')

    length = data.shape
    rows = length[0]

    id = [i for i in range(rows)]
    text = data[['text']]
    location = data[['location']]
    favorite = data[['favorite']]
    date = data[['date']]
    rt = data[['rt']]
    
    list_of_queries = []
    current = ''
    for x in range(rows):
        zero = int(id[x])
        one = str(text.at[x,'text'])
        two = str(location.at[x,'location'])
        three = str(favorite.at[x,'favorite'])
        four = str(date.at[x,'date'])
        five = str(rt.at[x,'rt'])

       
        list_of_queries.append("INSERT INTO Tweets (id, text, location, favorite, date, rt) VALUES ({0},'{1}','{2}','{3}','{4}', '{5}');".format(zero, one, two, three, four, five)) 

    f = open('insert/tweets_table_insert.sql', 'w')

    for x in list_of_queries:
        f.write(x+'\n')

    f.close()

def users_insert():
    '''
    create table Users
    (
	id SERIAL PRIMARY KEY NOT NULL,
	name VARCHAR NOT NULL,
	screen_name VARCHAR UNIQUE NOT NULL,
	geo_enabled BOOLEAN NOT NULL,
	verified BOOLEAN NOT NULL
    )
    '''
    #syntax for insert
    '''
    INSERT INTO TABLE_NAME (column1, column2, column3,...columnN)
    VALUES (value1, value2, value3,...valueN);
    '''
    data = pd.read_csv('cleaned_data/user.csv')

    length = data.shape
    rows = length[0]

    id = [i for i in range(rows)]
    name = data[['name']]
    screen_name = data[['screen_name']]
    geo_enabled = data[['geo_enabled']]
    verified = data[['verified']]

    list_of_queries = []

    for x in range(rows):
        zero = int(id[x])
        one = str(name.at[x,'name'])
        two = str(screen_name.at[x, 'screen_name'])
        three = str(geo_enabled.at[x, 'geo_enabled'])
        four  = str(verified.at[x, 'verified'])
        
        list_of_queries.append("INSERT INTO Type (id, name, screen_name, geo_enabled, verified) VALUES ({0}, '{1}', '{2}', '{3}', '{4}');".format(zero, str(one), two, three, four))

    f = open('insert/users_insert.sql', 'w')

    for x in list_of_queries:
        f.write(x+'\n')

    f.close()

def flu_insert():
    '''
   create table Flu	
    (
	flu_id SERIAL PRIMARY KEY NOT NULL,
	type VARCHAR NOT NULL
    )   
    '''
    #syntax for insert
    '''
    INSERT INTO TABLE_NAME (column1, column2, column3,...columnN)
    VALUES (value1, value2, value3,...valueN);
    '''

    data = pd.read_csv('cleaned_data/flu.csv')

    length = data.shape
    rows = length[0]

    flu_id = [i for i in range(rows)]
    type_ = data[['type']]

    list_of_queries = []
    for x in range(rows):
        zero = int(flu_id[x])
        one = str(type_.at[x,'type'])

        list_of_queries.append("INSERT INTO Flu (flu_id, type) VALUES ({0},'{1}');".format(zero, str(one)))

    f = open('insert/flu_insert.sql', 'w')

    for x in list_of_queries:
        f.write(x+'\n')

    f.close()


def symptoms_insert():
    '''
    create table Symptoms
    (
	symptoms_id SERIAL PRIMARY KEY NOT NULL,
	name VARCHAR NOT NULL,
	description VARCHAR UNIQUE NOT NULL
    )
    '''
    #syntax for insert
    
    '''
    INSERT INTO TABLE_NAME (column1, column2, column3,...columnN)
    VALUES (value1, value2, value3,...valueN);
    '''
    data = pd.read_csv('cleaned_data/symptoms.csv')

    length = data.shape
    rows = length[0]

    symptoms_id = [i for i in range(rows)]
    name = data[['name']]
    description = data[['description']]

    list_of_queries = []
    for x in range(rows):
        zero = int(symptoms_id[x])
        one = str(name.at[x,'name'])
        two = str(description.at[x, 'description'])
    
        list_of_queries.append("INSERT INTO Symptoms (symptoms_id, name, description) VALUES ({0},'{1}','{2}');".format(zero, one, two))

    f = open('insert/symptoms_insert.sql', 'w')

    for x in list_of_queries:
        f.write(x+'\n')

    f.close()

# tweets_table_insert()
# users_insert()
# flu_insert()
symptoms_insert()
 
