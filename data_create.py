import pandas as pd
df = pd.read_json('tweets_DM.json', lines=True)  # 使用 lines=True 來處理每行一個 JSON 物件的情況
#print(df)
data_identification = pd.read_csv('data_identification.csv')
data_emotion = pd.read_csv('emotion.csv')
#print(data_identification['identification'].value_counts())
#print(data_identification)
src = df['_source']
df_src = pd.DataFrame([{
    'hashtags': item['tweet']['hashtags'],
    'tweet_id': item['tweet']['tweet_id'],
    'text': item['tweet']['text']
} for item in src])
tweet_df = pd.concat([df_src, df['_score']], axis=1)
#print(df_merged)
#print(df_merged['identification'].value_counts())
df_result = pd.merge(tweet_df, data_identification, on='tweet_id')
#print(df_result)
df_train = df_result[df_result['identification'] == 'train']
df_test = df_result[df_result['identification'] == 'test']
df_train = pd.merge(df_train, data_emotion, on='tweet_id')

df_train = df_train[['tweet_id', 'text', 'hashtags', '_score', 'emotion']]
df_test = df_test[['tweet_id', 'text', 'hashtags', '_score']]

df_train.to_csv('train_data.csv', index=False)
df_test.to_csv('test_data.csv', index=False)

print(df_train)
print(df_test)

