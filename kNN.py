"""
最初は参考資料のコピー
"""
from sklearn.datasets import load_iris
iris= load_iris() # irisデータ取得
X = iris.data     # 説明変数(クラス推定用変数)
Y = iris.target   # 目的変数(クラス値)

# irisのデータをDataFrameに変換
iris_data = DataFrame(X, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
iris_target = DataFrame(Y, columns=['Species'])

# iris_targetが0〜2の値で分かりづらいので、あやめの名前に変換
def flower(num):
"""名前変換用関数"""
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Veriscolour'
    else:
        return 'Virginica'

iris_target['Species'] = iris_target['Species'].apply(flower)
iris = pd.concat([iris_data, iris_target], axis=1)

#データの概要
iris.head()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

sns.pairplot(iris, hue = 'Species', size =2) # hue:指定したデータで分割

#実際にkNNを実行してみる
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split # trainとtest分割用

# train用とtest用のデータ用意。test_sizeでテスト用データの割合を指定。random_stateはseed値を適当にセット。
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state=3) 

knn = KNeighborsClassifier(n_neighbors=6) # インスタンス生成。n_neighbors:Kの数
knn.fit(X_train, Y_train)                 # モデル作成実行
Y_pred = knn.predict(X_test)              # 予測実行

# 精度確認用のライブラリインポートと実行
from sklearn import metrics
metrics.accuracy_score(Y_test, Y_pred)    # 予測精度計測
