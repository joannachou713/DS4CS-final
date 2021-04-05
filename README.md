# Data Science for Cyber Security Final Project
Reference Paper
* [Identifying Suspicious URLs: An Application of Large-Scale Online Learning](http://cseweb.ucsd.edu/~jtma/papers/url-icml2009.pdf)
* [Beyond Blacklists: Learning to Detect Malicious Web Sites
from Suspicious URLs](http://cseweb.ucsd.edu/~jtma/papers/beyondbl-kdd2009.pdf)
### 資料集介紹
* 欄位儲存的是Lexical與host-based feature
    * Lexical Feature: 62%
    * Host-based Feature: 38%
* 論文作者研發出一項可real-time擷取url Feature的系統，而且此資料集已經經過作者們的前處理
    * Lexical: bag-of-words representation of tokens in the URL, where ‘/’, ‘?’, ‘.’, ‘=’, ‘-’, and ‘ ’ are delimiters
    * Host-based: record **where** the sites are hosted, **who** owns them, and  **how** they are managed
        * WHOIS information: records domain registration information
        * Location
        * Connection Speed
        * DNS-related properties(such as TTL)
* 不會有網頁內容相關的資訊
    1. 相對安全
    2. 只截取url資訊的運算較為輕量
* 欄位標籤為數字

# 使用DataFrame
## 資料讀取
* 擷取FeatureTypes
     ``` 
     [4, 5, 6, 8, 11, 16, 17, 18, 19, 21, 22, 23, 30, 33, 35, 39, 41, 43, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 120, 126, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 161, 194, 270, 7801]
     ```
* 讀取所有資料
    因為資料格式都是`.svm`檔格式，因此我使用以下函式來打開各個檔案，並把這121個檔案裡存的資料記在X_files, y_files的陣列裡，而陣列中的每筆當日資料型態為`scipy.sparse.csr_matrix`
    ```python=
    # Get `.svm` file content
    from joblib import Memory
    from sklearn.datasets import load_svmlight_file
    mem = Memory("./mycache")

    # get .svm
    @mem.cache
    def get_data(path):
        data = load_svmlight_file(path)
        return data[0], data[1]

    # get all .svm files
    X_files = []
    y_files = []
    for i in range(121):
      X, y = get_data(path=f"url_svmlight/Day{i}.svm")
      X_files.append(X)
      y_files.append(y)
    ```
## Sampling
* 轉為DataFrame
    在使用Pandas的DataFrame時，因為欄位總共有三百萬個，會導致RAM容易就被耗盡，所以我先選擇了`FeatureTypes`裡紀錄的欄位來進行分析及預測。
    ```python=
    # csr_matrix只擷取特定Columns
    # flist為FeatureType讀出的內容陣列
    # 只選取FeatureType中列出的real-valued features
    # 如果選取全部的欄位再轉為DataFrame會導致記憶體不足
    def getSingleDay(svm, flist):
        X = svm[:, flist]
        df = pd.DataFrame(X.toarray())
        return df

    df = pd.DataFrame()
    for svm in X_files:
        df = pd.concat([df, getSingleDay(svm, flist)])
    ```
* Sampling
    我先將讀出的X與y合併在一起，再進行採樣。其中y的部份因為+1代表惡意連結，-1代表非惡意，我將y使用LabelEncoder進行轉換，使+1為惡意連結，非惡意則以0表示。
    為了確保採樣結果的malicious與benign url比例為1:1，我把整個DataFrame切為malicious與benign意兩個小的資料表，再於這兩個小的資料表各取原資料筆數的0.5%，也就是11980。最後再將這兩個採樣結果合併並且打散。
    ```python=
    # Sampling
    le = LabelEncoder()
    df['target'] = le.fit_transform(pd.Series(np.concatenate(y_files)))
    benign = df.loc[df['target']==0]
    malicious = df.loc[df['target']==1]

    n_samples = int(df.shape[0]*0.005)
    benign = benign.sample(n=n_samples, random_state=1)
    malicious = malicious.sample(n=n_samples, random_state=1)
    df_sample = pd.concat([benign, malicious])

    print(benign.shape, malicious.shape)
    df_sample.head()
    ```
## EDA
* 各欄位值分佈
    ```python=
    df.hist(bins=50, figsize=(20,20))
    plt.show()
    ```
    ![](https://i.imgur.com/1nbmyD3.png)

* Correlation
    ```python=
    corr = df.corr()
    plt.figure(figsize=(10,10))      # Sample figsize in inches
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            annot=False,
            cmap="YlGnBu",
            linewidths=0.1,
            linecolor='white')
    ```
    ![](https://i.imgur.com/ipKdNpX.png)
    由Correlation Heatmap可以知道有些欄位內的值全部相同，需要將這些欄位刪除。另外我認為這裡可以改善的地方是，將所有feature與y的correlation也畫入圖內，在進行欄位的選擇，效果可能會比較好。

## Filter Features
### MinMaxScaler
因為這個資料集大部分為語義的資料，我推測欄位紀錄的會是這筆資料「有沒有」特定字串；此外，在各欄位值的分佈圖裡，大部分的欄位值只有0或1，因此我選擇使用MinMaxScaler進行標準化。
```python=
from sklearn.preprocessing import MinMaxScaler
df_normalized = pd.DataFrame(MinMaxScaler().fit_transform(df), columns = df.columns, index = df.index)
```
### Convert to Discrete
接下來我以標準化的結果，將所有欄位轉為離散型態，並將門檻設為0.5。也就是只要值小於0.5，新值就會被設定為0，大於0.5則新值為1。
```python=
# Threshold = 0.5
df_discrete = df_normalized
for column in df_discrete.columns:
  df_discrete[column] = df_discrete[column].apply(lambda x: 0 if x < 0.5 else 1)
```
### Delete the Columns Contain Same Values and Based on Information Gain
過濾欄位總和非0的欄位
```python=
not_zeros_discrete = [i for i, x in enumerate(df_discrete.sum()) if x!=0.0]
df_discrete = df_discrete[:][not_zeros_discrete]
```
使用Information Gain做過濾。我將Threshold設為0.0001，將亂度低於這個值的欄位刪除。
```python=
from sklearn.feature_selection import mutual_info_classif
drop_cols = []
for i, e in zip(df_discrete.columns, 
                    mutual_info_classif(df_discrete, y, discrete_features=True)):
  if e < 0.0001:
    print(i)
    drop_cols.append(i)
```
結果
```
7, 10, 12, 13, 15, 16, 20, 30, 33, 40, 48, 49, 60, 63
```
刪除資訊不足的欄位
```
df_discrete_entropy = df_discrete.copy().drop(drop_cols, axis=1)
```

### Dimentsion Reduction
* pca
    接下來我用處理過的DataFrame進行PCA的轉換
    ```python=
    from sklearn.decomposition import PCA
    pca = PCA()
    T = pca.fit_transform(df_discrete_entropy)
    ```
    ![](https://i.imgur.com/AEgfopE.png)
    
* lda
    ```python=
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(T[:, :10], y)
    ```
    ![](https://i.imgur.com/lKBReYt.png)
    結果好像也沒辦法準確的分類
* t-sne
    ```python=
    # X: df_discrete_entropy
    from sklearn.manifold import TSNE
    X_embedded = TSNE().fit_transform(df_discrete_entropy)
    X_embedded.shape
    ```
## 切割資料集(Train/Test)
我決定將三次降維的結果都進行訓練與預測
```python=
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(T[:, :4], y, test_size=0.33, random_state=42)
X_train_lda, X_test_lda, y_train_lda, y_test_lda = train_test_split(X_lda, y, test_size=0.33, random_state=42)
X_train_embedded, X_test_embedded, y_train_embedded, y_test_embedded = train_test_split(X_embedded, y, test_size=0.33, random_state=42)
```
## 選用模型
* SVM
* Decision Tree
* Logistic Regression
* Neural Network
```python=
svm = svm.SVC()
svm.fit(X_train, y_train)

dt = tree.DecisionTreeClassifier(max_depth = 5)
dt.fit(X_train, y_train)

lr = LogisticRegression(random_state=0).fit(X_train, y_train)

model = keras.Sequential([
    Dense(50, activation=tf.nn.relu, input_shape=(1, 50)),
    Dense(50, activation=tf.nn.relu),
    Dense(50, activation=tf.nn.relu),
    Dense(50, activation=tf.nn.relu),
    Dropout(0.25),
    Dense(50, activation=tf.nn.relu),
    Dense(50, activation=tf.nn.relu),
    Dense(20, activation=tf.nn.relu),
    Dropout(0.25),
    Dense(50, activation=tf.nn.relu),
    Dense(50, activation=tf.nn.relu),
    Dense(20, activation=tf.nn.relu),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
#防止接近最低點時的左右振盪
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
model.fit(X_train, y_train, epochs=100, verbose=2, validation_split = 0.1,
                    callbacks=early_stop) #callback: 每輪(epoch)做完要做的事
```
## 訓練結果
* Cross Entropy
* Accuracy
* F1 score
* Confusion Matrix

### PCA
#### Scores
|       Model        |        SVM         |   Decision Tree    | Logistic Regression  |   Neural Network   |
|--------------------|--------------------|--------------------|----------------------|--------------------|
|   Cross Entropy    |       9.903        |       10.58        |         11.91        |        8.78        |
|      Accuracy      |       0.7133       |       0.6937       |         0.6552       |       0.7458       |
|      F1 Score      |       0.6826       |       0.6797       |         0.6593       |       0.7425       |
#### Confusion Matrix
![](https://i.imgur.com/2MLyice.png)
根據Confusion Matrix，以預防連線到惡意連結而言，SVM與Neural Network的分類效果較差，容易將實際上是惡意的連結歸類為非惡意的，而Decision Tree與Logistic Regression則相對較能夠防止用戶連上惡意連結。
### LDA
#### Score
|       Model        |        SVM         |   Decision Tree    | Logistic Regression  |   Neural Network   |
|--------------------|--------------------|--------------------|----------------------|--------------------|
|   Cross Entropy    |       10.59        |       9.461        |         10.42        |       9.575        |
|      Accuracy      |       0.6933       |       0.7261       |         0.6982       |       0.7228       |
|      F1 Score      |       0.6977       |       0.7054       |         0.6862       |       0.7027       |
#### Confusion Matrix
![](https://i.imgur.com/5mJkRNk.png)
由此圖可知，在所有選用的模型裡，LDA轉換過再進行訓練的結果比PCA轉換再訓練的結果更差，更難以準確的預防使用者連上惡意的連結。
### T-SNE
#### Score
|       Model        |        SVM         |   Decision Tree    | Logistic Regression  |   Neural Network   |
|--------------------|--------------------|--------------------|----------------------|--------------------|
|   Cross Entropy    |       11.08        |       10.43        |          13.1        |       8.915        |
|      Accuracy      |       0.6791       |       0.698        |         0.6208       |       0.7419       |
|      F1 Score      |       0.6817       |       0.6896       |         0.6096       |       0.739        |
#### Confusion Matrix
![](https://i.imgur.com/x1jIn5Q.png)
使用T-SNE進行維度轉換的結果更糟，Logistic Regression的準確度甚至不到六成。
# 使用csr_matrix
因為欄位過多導致無法將完整的資料集從`csr_matrix`轉換為`DataFrame`，而且在閱讀完資料集作者所撰寫的論文後，我發現這筆資料集已經經過預處理（如bag-of-words等），因此我想嘗試不做欄位內值的轉換，直接以`csr_matrix`的型態進行抽樣，並使用降維轉換整筆資料集，藉以提高準確度。
```python=
# Sampling and set X and y
sample_idx = np.arange(x_csr.shape[0])
np.random.shuffle(sample_idx)
sample_idx = sample_idx[:int(0.01*x_csr.shape[0])]
x_csr_sampled = x_csr[sample_idx]
y = y[sample_idx]
```
## Filter Features: Dimension Reduction
* TruncatedSVD
    `csr_matrix`版的PCA。因為在將`csr_matrix`倒入PCA轉換時報錯，經過搜尋之後得知`truncatedSVD`才能夠執行對`csr_matrix`矩陣的線性轉換。
    
    ```python=
    from sklearn.decomposition import TruncatedSVD
    # from sklearn.decomposition import PCA
    svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
    x_svd = svd.fit_transform(x_csr_sampled)
    ```
    解釋比例圖：
    ![](https://i.imgur.com/CUFEGYs.png)
    因為解釋變異比例只有52%，因此我選擇使用所有50個component進行訓練與預測

## 切割資料集(Train/Test)
```python=
from sklearn.model_selection import train_test_split
X_train_svd, X_test_svd, y_train_svd, y_test_svd = train_test_split(df_svd, y, test_size=0.33, random_state=42) 
```
## 選用模型
* SVM
* Decision Tree
* Logistic Regression
* Neural Network

## 訓練結果
* Cross Entropy
* Accuracy
* F1 score
* Confusion Matrix
### Score
|       Model        |        SVM         |   Decision Tree    | Logistic Regression  |   Neural Network   |
|--------------------|--------------------|--------------------|----------------------|--------------------|
|   Cross Entropy    |       1.468        |       2.813        |         1.468        |       1.385        |
|      Accuracy      |       0.9575       |       0.9186       |         0.9575       |       0.9599       |
|      F1 Score      |       0.9365       |       0.8747       |         0.9359       |       0.9393       |
### Confusion Matrix
![](https://i.imgur.com/sfyymjW.png)


# 結果比較
| Model         | PCA Neural Network | LDA Neural Network | T-SNE Neural Network | SVD Neural Network |
| ------------- | ------------------ | ------------------ | -------------------- | ------------------ |
| Cross Entropy | 8.78               |  9.575             | 8.915                | 1.385              |
| Accuracy      | 0.7458             |  0.7228            | 0.7419               | 0.9599             |
| F1 Score      | 0.7425             |  0.7027            | 0.739                | 0.9393             |

![](https://i.imgur.com/PtoxNn7.png)<br/>
最後將所有模型的準確度與Confusion Matrix相互比較，發現只經過truncatedSVD轉換的資料所訓練出來的結果是最好的，各項準確度都高達90%以上。我推測是因為論文作者有針對文字資料，以bag-of-words等方式進行過前處理，所以直接進行降維送入模型訓練的結果才會是最好的。
