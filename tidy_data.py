# 1
# Attendance Data
# Read the data from the attendance table and calculate an attendance percentage for each student. 
# One half day is worth 50% of a full day, and 10 tardies is equal to one absence.
# You should end up with something like this:

'''
name
Billy    0.5250
Jane     0.6875
John     0.9125
Sally    0.7625
Name: grade, dtype: float64
'''

from env import get_db_url
import env
import pandas as pd

url = get_db_url('tidy_data')
sql = 'select * from attendance'

attendance_df = pd.read_sql(sql, url)
#attendance_df
attendance_df = attendance_df.rename(columns={'Unnamed: 0':'first_name'})
attendance_df = attendance_df.melt(id_vars='first_name', var_name='date', value_name='attend')
attendance_df.attend = attendance_df.attend.map({'P':1,'A':0,'T':.9,'H':.5})


attendance_df.groupby('first_name').mean()


# 2
# Coffee Levels
# Read the coffee_levels table.
# Transform the data so that each carafe is in it's own column.
# Is this the best shape for the data?

sql = 'select * from coffee_levels'

coffee_df = pd.read_sql(sql, url)
# coffee_df
coffee_df = coffee_df.reindex(['coffee_carafe', 'hour', 'coffee_amount'], axis=1)


# pivot table for distinct carafe columns
coffee_df = coffee_df.pivot_table(index='hour', columns='coffee_carafe')
# The above is an example of untidy data. the var type'coffee_carafe' is 
# differing but of the same category therefore not needing seperate columns
coffee_df = coffee_df.melt(ignore_index=False,col_level=1,var_name='coffee_carafe',value_name='coffee_amount').reset_index()
# reset table with above



# 3
# Cake Recipes
# Read the cake_recipes table. This data set contains cake tastiness 
# scores for combinations of different recipes, oven rack positions, 
# and oven temperatures.

# Read the cake_recipes table
sql = 'select * from cake_recipes'
cake_df = pd.read_sql(sql, url)

# tidy the data as necc.
cake_df = cake_df.melt(id_vars='recipe:position', var_name='temp', value_name='score')
cake_df[['recipe','position']] = cake_df['recipe:position'].str.split(':', expand=True)
cake_df = cake_df.drop(columns='recipe:position')

# Which recipe, on average, is the best?
cake_df.groupby('recipe').mean()
print('best average tasty cake:{} {}'.format((cake_df.groupby('recipe').mean().max()),(cake_df.groupby('recipe').mean().idxmax())))

# Which oven temperature, on average, produces the best results?
cake_df.groupby('temp').mean()
print('best avg temp for tastiness of cake: {}'.format(cake_df.groupby('temp').mean().idxmax()))


# Which combination of recipe, rack position, and temperature gives the best result?
cake_df.groupby(['temp','recipe','position']).mean().idxmax()





