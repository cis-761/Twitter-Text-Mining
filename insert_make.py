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

       
        list_of_queries.append("INSERT INTO Tweeets (id, text, location, favorite, date, rt) VALUES ({0},'{1}','{2}','{3}','{4}', '{5}');".format(zero, one, two, three, four, five)) 

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

def pokemon_moves_insert():
    '''
    moveNum int not null,
    name varchar(30) not null,
    '''
    #syntax for insert
    '''
    INSERT INTO TABLE_NAME (column1, column2, column3,...columnN)
    VALUES (value1, value2, value3,...valueN);
    '''

    data = pd.read_csv('final_clean_data/moves_official.csv')

    moveNum = data[['move_num']]
    name = data[['name']]
    list_of_queries = []
    for x in range(251):
        zero = int(moveNum.at[x,'move_num'])
        one = str(name.at[x,'name'])
        #two = str(description.at[x,'Classification'])
        list_of_queries.append("INSERT INTO Moves (moveNum, name) VALUES ({0},'{1}');".format(zero, str(one)))

    f = open('insert/move_table_insert.sql', 'w')

    for x in list_of_queries:
        f.write(x+'\n')

    f.close()


def poke_type_insert():
    '''
    pokeNumber int not null,
    typeNum int not null
    '''

    #syntax for insert
    '''
    INSERT INTO TABLE_NAME (column1, column2, column3,...columnN)
    VALUES (value1, value2, value3,...valueN);
    '''
    data = pd.read_csv('final_clean_data/pokemon_type_official.csv')

    poke_num = data[['poke_num']]
    type_num = data[['type_num']]
    list_of_queries = []
    for x in range(47):
        zero = int(poke_num.at[x,'poke_num'])
        one = int(type_num.at[x,'type_num'])
        #two = str(description.at[x,'Classification'])
        list_of_queries.append("INSERT INTO PokeType (pokeNumber, typeNum) VALUES ({0},{1});".format(zero, one))

    f = open('insert/poke_type_insert.sql', 'w')

    for x in list_of_queries:
        f.write(x+'\n')

    f.close()

def level_up_moves_insert():
    '''
    pokeNumber int not null,
    moveNum int not null,
    level int not null,
    '''
    data = pd.read_csv('final_clean_data/level_up_official.csv')

    poke_num = data[['poke_num']]
    type_num = data[['level_up_move_num']]
    level = data[['level']]
    list_of_queries = []
    for x in range(1348):
        zero = int(poke_num.at[x,'poke_num'])
        one = int(type_num.at[x,'level_up_move_num'])
        two = int(level.at[x,'level'])
        #two = str(description.at[x,'Classification'])
        list_of_queries.append("INSERT INTO LevelUpMoves (pokeNumber, moveNum, level) VALUES ({0},{1},{2});".format(zero, one, two))

    f = open('insert/level_up_moves.sql', 'w')

    for x in list_of_queries:
        f.write(x+'\n')

    f.close()

def egg_moves_insert():
    '''
    create table EggMoves{
        pokeNumber int not null,
        moveNum int not null,
        primary key (pokeNumber, moveNum)
        foreign key (pokeNumber) references Pokemon(number)
        foreign key (moveNum) references Moves(moveNum)
    }
    '''
    data = pd.read_csv('final_clean_data/pokemon_egg_move_official.csv')

    poke_num = data[['poke_num']]
    egg_move = data[['egg_move_num']]

    list_of_queries = []
    for x in range(605):
        zero = int(poke_num.at[x,'poke_num'])
        one = int(egg_move.at[x,'egg_move_num'])

        #two = str(description.at[x,'Classification'])
        list_of_queries.append("INSERT INTO EggMoves (pokeNumber, moveNum) VALUES ({0},{1});".format(zero, one))

    f = open('insert/egg_moves_insert.sql', 'w')

    for x in list_of_queries:
        f.write(x+'\n')

    f.close()

def egg_group_insert():
    '''
    create table EggGroup {
    name varchar (30) not null,
    canBreed boolean not null,
    primary key (name)
    unique(name)
    }
    '''
    data = pd.read_csv('final_clean_data/egg_groups_official.csv')

    name = data[['name']]
    can_breed = data[['can_breed']]

    list_of_queries = []
    for x in range(14):
        zero = str(name.at[x,'name'])
        one = can_breed.at[x,'can_breed']

        #two = str(description.at[x,'Classification'])
        list_of_queries.append("INSERT INTO EggGroup(name, canBreed) VALUES ('{0}',{1});".format(zero, one))

    f = open('insert/egg_groups_insert.sql', 'w')

    for x in list_of_queries:
        f.write(x+'\n')

    f.close()


def poke_egg_group_insert():
    '''
    create table PokeEggGroup {
    pokeNumber int not null,
    name varchar(30) not null,
    primary key (pokeNumber, name)
    foreign key (pokeNumber) references Pokemon(number)
    foreign key (name) references EggGroup(name)
    }
    '''
    data = pd.read_csv('final_clean_data/pokemon_egg_group_official.csv')

    poke_num = data[['poke_num']]
    egg_group_name = data[['egg_group_name']]

    list_of_queries = []
    for x in range(188):
        zero = str(poke_num.at[x,'poke_num'])
        one = str(egg_group_name.at[x,'egg_group_name'])

        #two = str(description.at[x,'Classification'])
        list_of_queries.append("INSERT INTO PokeEggGroup(pokeNumber, name) VALUES ('{0}','{1}');".format(zero, one))

    f = open('insert/pokemon_egg_group.sql', 'w')

    for x in list_of_queries:
        f.write(x+'\n')

    f.close()

#tweets_table_insert()
users_insert()
# pokemon_moves_insert()
# poke_type_insert()
# level_up_moves_insert()
# egg_moves_insert()
# egg_group_insert()
# poke_egg_group_insert()
