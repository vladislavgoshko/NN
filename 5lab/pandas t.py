import pandas as pd
exel_data = pd.read_excel("lab5/data/Температура в море Рио-Де-Жанейро 1-9-2022–10-11-2022.xlsx")

df = exel_data.drop(labels=[0], axis = 0)

# for i in range(2, len(df.columns), 2):
#   index = len(df.columns[i-1])-1
#   newName = df.columns[i-1][index-3:index]
#   oldName = df.columns[i]
#   df.rename(columns={oldName : newName}, inplace = True)

course_df = df[['Дата', 't в море']]

course_df.head()