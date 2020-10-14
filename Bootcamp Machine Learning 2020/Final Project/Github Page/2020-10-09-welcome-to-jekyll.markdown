---
layout: post
title:  "FIFA 19 Data Analysis"
date:   2020-10-11 23:38 +0700
categories: jekyll update
---
# Analisis Data FIFA 19 dan Prediksi Overall Memakai Sckit-Learn dan Keras

### Introduction
FIFA 19 adalah sebuah game simulasi sepak bola yang dikembangkan oleh Electonics Arts. Game ini diperkenalkan pada tanggal 6  Juni 2018 pada event E3 2018. Game ini dirilis pada tanggal 28 September 2018 untuk konsol PlayStation 3, PlayStation 4, Xbox One, Nintendo Switch, dan Microsoft Windows. Di edisi kali ini adalah pertama kalinya fitur kompetisi UEFA termasuk UEFA Champions League, UEFA Europa League, dan UEFA Super Cup muncul di game FIFA 19 ini. FIFA 19 juga merupakan series game FIFA terakhir yang dirilis untuk konsol PlayStation 3.

Pada kesempatan kali ini saya akan melakukan analisis data untuk dataset FIFA 19. Data ini langsung diambil dari [Sofifa][sofifa], sebuah website yang berisi mengenai data lengkap dari para pemain serta tim yang ada di game FIFA 19. 

### Data Cleaning

Data yang dipakai dapat dilihat dibawah ini 

![26ZmQa.jpg](https://iili.io/26ZmQa.jpg)

![26ZyCJ.jpg](https://iili.io/26ZyCJ.jpg)

![26Zbhg.jpg](https://iili.io/26Zbhg.jpg)

![26ZDTF.jpg](https://iili.io/26ZDTF.jpg)

![26ZZp1.jpg](https://iili.io/26ZZp1.jpg)

![26ZQkP.jpg](https://iili.io/26ZQkP.jpg)

Dataset ini terdiri dari 18207 baris dan 89 kolom. Namun tidak semua 89 kolom akan dipakai, akan ada beberapa kolom yang akan kita *drop* dan kita ringkas. 

Dataset ini diupload ke dalam Google Colab dengan memakai code `pd.read_csv` yang akan membaca file csv menjadi sebuah DataFrame dan akan diberi nama `fifa19`

Kemudian kita akan cek jumlah missing values yang terdapat pada dataset ini dengan memakai code `fifa19.isnull().sum().sum()`. Pada dataset ini terdapat sekitar 77000 missing values. 

![26iEVR.jpg](https://iili.io/26iEVR.jpg)

Kemudian akan dicek kolom-kolom apa saja yang terdapat missing values dengan code `fifa19.isnull().sum().sort_values()`. Berikut ini list dari kolom yang terdapat missing values

![26PL6N.jpg](https://iili.io/26PL6N.jpg)   ![26PsGp.jpg](https://iili.io/26PsGp.jpg) 

![26sqfp.jpg](https://iili.io/26sqfp.jpg)   ![26sKsR.jpg](https://iili.io/26sKsR.jpg)
 
Kita akan mengisi missing values tersebut. Metode pengisiannya berbeda-beda, ada yang diisi dengan mean, ada yang diisi dengan nilai modusnya dan ada yang hanya akan diisi oleh nilai 0. Pengisian missing values ini memakai library pandas dengan code `fifa19.fillna()`. Untuk pengisian missing valuesnya dapat dilihat pada gambar dibawah ini

{% highlight ruby %}
fifa19['ShortPassing'].fillna(fifa19['ShortPassing'].mean(), inplace=True)
fifa19['Volleys'].fillna(fifa19['Volleys'].mean(), inplace=True)
fifa19['Dribbling'].fillna(fifa19['Dribbling'].mean(), inplace=True)
fifa19['Curve'].fillna(fifa19['Curve'].mean(), inplace=True)
fifa19['FKAccuracy'].fillna(fifa19['FKAccuracy'].mean(), inplace=True)
fifa19['LongPassing'].fillna(fifa19['LongPassing'].mean(), inplace=True)
fifa19['BallControl'].fillna(fifa19['BallControl'].mean(), inplace=True)
fifa19['HeadingAccuracy'].fillna(fifa19['HeadingAccuracy'].mean(), inplace=True)
fifa19['Finishing'].fillna(fifa19['Finishing'].mean(), inplace=True)
fifa19['Crossing'].fillna(fifa19['Crossing'].mean(), inplace=True)
fifa19['Weight'].fillna('200 lbs', inplace=True)
fifa19['Contract Valid Until'].fillna(2019, inplace=True)
fifa19['Height'].fillna("5'11", inplace=True)
fifa19['Loaned From'].fillna('None', inplace=True)
fifa19['Joined'].fillna('Jul 1, 2018', inplace=True)
fifa19['Jersey Number'].fillna(8, inplace=True)
fifa19['Body Type'].fillna('Normal', inplace=True)
fifa19['Position'].fillna('ST', inplace=True)
fifa19['Club'].fillna('No Club', inplace=True)
fifa19['Work Rate'].fillna('Medium/ Medium', inplace=True)
fifa19['Skill Moves'].fillna(fifa19['Skill Moves'].median(), inplace=True)
fifa19['Weak Foot'].fillna(3, inplace=True)
fifa19['Preferred Foot'].fillna('Right', inplace=True)
fifa19['International Reputation'].fillna(1, inplace=True)
fifa19['Wage'].fillna('€200K', inplace=True)
fifa19['Release Clause'].fillna('0', inplace=True)

fifa19.fillna(0, inplace=True)
{% endhighlight %}

Untuk memastikan, kita akan cek sekali lagi missing values yang terdapat pada dataset dengan code `fifa19.isnull().sum().sum()`

![26ilDJ.jpg](https://iili.io/26ilDJ.jpg)

Terlihat bahwa missing values yang tadinya berjumlah 77000, sudah kita hilangkan dan sekarang dataset sudah bersih dari missing values.

Kemudian kita akan drop beberapa feature seperti `'Unnamed: 0'`, `'Photo'`, `'Club Logo'`, `'Flag'` dengan code `fifa19.drop(['Unnamed: 0', 'Photo', 'Club Logo', 'Flag'], axis=1, inplace=True)` karena tidak memberikan informasi apapun terkait target yang ingin kita prediksi yaitu `Overall`.

Kemudian kita akan meringkas beberapa feature yang ada di dataset ini dengan membuat 6 feature baru yang berisi hasil rata-rata dari feature-feature yang sudah kita ringkas

{% highlight ruby %}
def pace(fifa):
  return int(round((fifa[['SprintSpeed', 'Acceleration']].mean()).mean()))

def passing(fifa):
  return int(round((fifa[['ShortPassing', 'Crossing', 'Vision', 'LongPassing', 'Curve', 'FKAccuracy']].mean()).mean()))

def defending(fifa):
  return int(round((fifa[['StandingTackle', 'Marking', 'Interceptions', 'HeadingAccuracy', 'SlidingTackle']].mean()).mean()))

def shooting(fifa):
  return int(round((fifa[['Finishing', 'LongShots', 'ShotPower', 'Volleys', 'Positioning', 'Penalties']].mean()).mean()))

def dribbling(fifa):
  return int(round((fifa[['Dribbling', 'BallControl', 'Agility', 'Balance']].mean()).mean()))

def physical(fifa):
  return int(round((fifa[['Strength', 'Stamina', 'Aggression', 'Jumping']].mean()).mean()))
{% endhighlight %}

Lalu, keenam feature ini akan ditambahkan kedalam dataset dan feature-feature yang tadi dipakai untuk membuat 6 feature baru ini akan di drop. 

{% highlight ruby %}
fifa19['Pace'] = fifa19.apply(pace, axis=1)
fifa19['Passing'] = fifa19.apply(passing, axis=1)
fifa19['Defending'] = fifa19.apply(defending, axis=1)
fifa19['Shooting'] = fifa19.apply(shooting, axis=1)
fifa19['Dribbling'] = fifa19.apply(dribbling, axis=1)
fifa19['Physical'] = fifa19.apply(physical, axis=1)

fifa19.drop(['SprintSpeed', 'Acceleration', 'ShortPassing', 'Crossing', 'Vision', 'LongPassing', 'Curve', 'FKAccuracy', 'StandingTackle', 'Marking', 'Interceptions', 
             'HeadingAccuracy', 'SlidingTackle', 'Finishing', 'LongShots', 'ShotPower', 'Volleys', 'Positioning', 'Penalties', 'BallControl', 'Agility', 'Balance', 
             'Strength', 'Stamina', 'Aggression', 'Jumping'], axis=1, inplace=True)
{% endhighlight %}

Berikutnya kita akan cek dimensi dari dataset kita dengan code `fifa19.shape`

![26suzG.jpg](https://iili.io/26suzG.jpg)

Terlihat bahwa kolom dataset kita yang tadinya ada 89, sekarang menjadi 64 karena ada beberapa feature yang di drop dan kita ringkas.

Berikutnya kita akan melakukan data cleaning pada kolom `weight`,  `value`, `wage`, dan `release clause`.
Untuk kolom `weight` kita akan menghilangkan tulisan **lbs** setelah nominal berat badannya agar tipe data untuk kolom `weight` bisa diubah dari yang tadinya *object* menjadi *float*. Berikut proses data cleaningnya.

{% highlight ruby %}
def weight_cleaning(weight):
  out = weight.replace('lbs', '')
  return float(out)

fifa19['Weight'] = fifa19['Weight'].apply(lambda x : weight_cleaning(x))
{% endhighlight %}

Untuk kolom `value`, `wage`, dan `release clause` kita akan menghilangkan simbol **€**, dan simbol **M** atau **K** yang berada disebelum dan setelah nominalnya. Tujuannya juga agar tipe data dari ketiga feature tersebut dapat kita casting dari *object* menjadi **float**. Berikut proses data cleaningnya.

{% highlight ruby %}
def wage_cleaning(wage):
  out = wage.replace('€', '')
  if 'M' in out:
    out = float(out.replace('M', ''))*1000000
  elif 'K' in out:
    out = float(out.replace('K', ''))*1000
  return float(out)

fifa19['Value'] = fifa19['Value'].apply(lambda x : wage_cleaning(x))
fifa19['Wage'] = fifa19['Wage'].apply(lambda x : wage_cleaning(x))
fifa19['Release Clause'] = fifa19['Release Clause'].apply(lambda x : wage_cleaning(x))
{% endhighlight %}

### Data Exploration

Setelah datasetnya kita cleaning, berikutnya kita akan melihat insight-insight yang bisa didapat dari dataset ini.
Pertama dibuat fungsi untuk melihat list pemain-pemain yang berasal dari suatu negara. Saya ambil contoh pemain-pemain yang berasal dari negara Italia. Untuk prosesnya dapat dilihat pada gambar berikut.

{% highlight ruby %}
def country(x):
  return fifa19[fifa19['Nationality'] == x][['Name', 'Overall', 'Potential', 'Position']]

country('Italy')
{% endhighlight %}

Berikutnya kita membuat fungsi untuk menampilkan list dari skuad dari tim-tim tertentu yang ada di dataset FIFA 19 ini. Saya ambil contoh list dari skuad Milan. Prosesnya dapat dilihat pada gambar berikut.

{% highlight ruby %}
def club(x):
  return fifa19[fifa19['Club'] == x][['Name', 'Jersey Number', 'Position', 'Overall', 'Nationality', 'Age', 'Wage', 'Value', 'Contract Valid Until']]

club('Milan')
{% endhighlight %}

Berikutnya kita akan membuat visualisasi untuk mendapatkan insight-insight dari dataset. 

Kita akan melihat **Preferred Foot** dari para pesepakbola yang ada di FIFA 19. Dari gambar dibawah terlihat bahwa mayoritas pesepakbola di FIFA 19 menggunakan kaki kanan sebagai kaki dominan mereka.

![266vmG.jpg](https://iili.io/266vmG.jpg)

Kita akan melihat **International Reputation** para pesepakbola di FIFA 19. Terlihat bahwa mayoritas pesepakbola di dataset memiliki *international reputation* bernilai 1 yang berarti mayoritas pesepakbola di FIFA 19 belum memiliki pengalaman bermain bersama tim nasionalnya. Semakin tinggi nilainya, maka menunjukkan pemain tersebut sudah memiliki pengalaman yang bagus bersama tim nasionalnya dan semakin mudah pemain tersebut untuk dilirik oleh klub-klub besar.

![266L7e.jpg](https://iili.io/266L7e.jpg)

Kita akan melihat **Weak Foot** dari pesepakbola di FIFA 19. Terlihat bahwa mayoritas pesepakbola di FIFA 19 memiliki nilai *weak foot* 3 yang berarti mayoritas pesepakbola di FIFA 19 memiliki kemampuan yang baik menendang dengan kaki yang bukan kaki dominannya. Semakin tinggi nilainya berarti pesepakbola tersebut memiliki kemampuan menendang dengan kaki yang bukan dominannya sama baiknya dengan kaki dominannya. Begitu pula sebaliknya.

![26PMve.jpg](https://iili.io/26PMve.jpg)

Kita akan melihat distribusi pesepakbola ditinjau berdasarkan **Position**nya. Dari 27 jenis posisi yang ada di FIFA 19, mayoritas pesepakbola di FIFA 19 merupakan pesepakbola yang berposisi sebagai Striker (ST)

![26PE37.jpg](https://iili.io/26PE37.jpg)

Kita akan meilihat distribusi gaji pesepakbola di FIFA 19. Terlihat bahwa mayoritas pesepakbola di FIFA 19 memiliki gaji dibawah €260K dan hanya beberapa pesepakbola saja yang memiliki gaji diatas €260K per minggunya

![26PlG2.jpg](https://iili.io/26PlG2.jpg)

Kita akan melihat jenis **Work Rate** pesepakbola di FIFA 19. Terlihat bahwa mayoritas pesepakbola di FIFA 19 memiliki jenis *work rate* Medium/Medium yang berarti pesepakbola tersebut memiliki usaha yang cukup dalam bertahan dan juga menyerang. Sebagai perbandingan, pesepakbola yang memiliki *work rate* High/Low berarti memiliki usaha yang buruk dalam bertahan, namun dia akan sangat bekerja keras dalam melakukan penyerangan.

![26P04S.jpg](https://iili.io/26P04S.jpg)

Kita akan melihat 25 negara dengan pesepakbola terbanyak di FIFA 19. Terlihat bahwa mayoritas pesepakbola di FIFA 19 berasal dari Inggris.

![26PGa9.jpg](https://iili.io/26PGa9.jpg)

Kita akan melihat *overall* dari para pesepakbola di FIFA 19 berdasarkan asal negaranya. Terlihat bahwa pemain yang berasal dari Brasil dan Spanyol memiliki rata-rata *overall* tertinggi dibanding negara lainnya.

![26Phjj.jpg](https://iili.io/26Phjj.jpg)

Kita akan melihat distribusi gaji pesepakbola di FIFA 19 berdasarkan asal negaranya. Disini kita akan mengambil 10 negara teratas dengan jumlah pesepakbola terbanyak. Dari 10 negara tersebut, terlihat bahwa pemain-pemain yang berasal dari Brasil memiliki rata-rata gaji tertinggi dibanding negara lainnya.

![26PXTb.jpg](https://iili.io/26PXTb.jpg)

Setelah dilihat distribusi gaji berdasarkan asal negaranya, sekarang akan dilihat distribusi gaji berdasarkan klubnya. Disini akan diambil 10 klub dengan gaji tertinggi di FIFA 19. Terlihat bahwa Real Madrid merupakan klub yang memiliki rerata gaji tertinggi di FIFA 19.

![26PVyu.jpg](https://iili.io/26PVyu.jpg)

Kita akan melihat pesepakbola terbaik di tiap posisi berdasarkan *Potential* dan *Overall* scoresnya. Untuk code dan beberapa outputnya dapat dilihat dibawah ini 

{% highlight ruby %}
fifa19.iloc[fifa19.groupby(fifa19['Position'])['Potential'].idxmax()][['Position', 'Name', 'Club', 'Age', 'Nationality', 'Potential']] #Untuk dilihat berdasarkan overall, cukup ganti kolom 'Potential' menjadi 'Overall'
{% endhighlight %}

![26t4wX.jpg](https://iili.io/26t4wX.jpg)

![26t6tn.jpg](https://iili.io/26t6tn.jpg)

Berikutnya kita akan melihat pesepakbola dengan *Overall* dan *Potential* terendah di tiap posisinya. Untuk code dan outputnya dapat dilihat dibawah ini.

{% highlight ruby %}
fifa19.iloc[fifa19.groupby(fifa19['Position'])['Potential'].idxmin()][['Position', 'Name', 'Club', 'Age', 'Nationality', 'Potential']] #Untuk dilihat berdasarkan overall, cukup ganti kolom 'Potential' menjadi 'Overall'
{% endhighlight %}

![26tLPf.jpg](https://iili.io/26tLPf.jpg)

![26tsMG.jpg](https://iili.io/26tsMG.jpg)

Kita akan melihat 10 pesepakbola dengan usia paling muda dan paling tua di FIFA 19. Untuk code dan outputnya dapat dilihat dibawah ini.

{% highlight ruby %}
termuda = fifa19.sort_values('Age', ascending=True)[['Name', 'Age', 'Club', 'Nationality']].head(10)
#untuk mengetahui usia paling tua cukup ganti menjadi ascending=False
{% endhighlight %}

![26ttcl.jpg](https://iili.io/26ttcl.jpg)

![26tDS2.jpg](https://iili.io/26tDS2.jpg)

Terlihat bahwa O. Pérez yang bermain untuk klub Pachuca merupakan pemain tertua di FIFA 19 dengan usia 45 tahun. Sementara itu terdapat beberapa pemain yang berusia 16 tahun yang merupakan usia termuda di FIFA 19.

Lalu, kita akan melihat pesepakbola dengan masa bakti paling lama dan paling sebentar untuk sebuah klub. Untuk mengetahuinya kita akan mengambil tahun yang terdapat pada kolom `Joined`, kemudian setelah itu kita akan melakukan operasi pengurangan dari tahun saat ini (2020) dengan tahun yang sudah kita peroleh dari kolom `Joined` tadi. Untuk code dan outputnya dapat dilihat dibawah ini.

{% highlight ruby %}
import datetime
now = datetime.datetime.now()
fifa19['Join_year'] = fifa19['Joined'].dropna().map(lambda x : x.split(',')[1].split(' ')[1]) #kita akan ambil tahunnya saja
fifa19['Years_of_member'] = (fifa19['Join_year'].dropna().map(lambda x : now.year - int(x))).astype('int')
masa_bakti_panjang = fifa19[['Name', 'Club', 'Years_of_member']].sort_values(by='Years_of_member', ascending=False).head(10)
#untuk mengetahui masa bakti paling sebentar, cukup ganti menjadi ascending=True
{% endhighlight %}

![26DJou.jpg](https://iili.io/26DJou.jpg)

![26D9te.jpg](https://iili.io/26D9te.jpg)

Terlihat bahwa O. Pérez yang bermain untuk klub Pachuca merupakan pemain dengan masa bakti terlama yakni 29 tahun. Sementara itu terdapat beberapa pemain dengan masa bakti paling sebentar untuk sebuah klub, yakni hanya 2 tahun.

Kita akan melihat pesepakbola dengan kaki dominan kanan dan kiri terbaik di FIFA 19. Untuk code dan outputnya dapat dilihat dibawah ini.

{% highlight ruby %}
fifa19[fifa19['Preferred Foot'] == 'Right'][['Name', 'Age', 'Club', 'Nationality']].head(10)
#untuk melihat pesepakbola dengan kaki dominan kanan, cukup ganti menjadi Preferred Foot = Left
{% endhighlight %}

![26D2Pj.jpg](https://iili.io/26D2Pj.jpg)

![26DdMb.jpg](https://iili.io/26DdMb.jpg)

Terlihat bahwa Lionel Messi menjadi pemain dengan kaki kiri dominan terbaik dan Cristiano Ronaldo menjadi pemain dengan kaki dominan kanan terbaik.

Kita akan melihat klub dengan jumlah pemain yang berasal dari negara yang berbeda dengan jumlah terbanyak dan paling sedikit. Untuk code dan outputnya dapat dilihat dibawah ini.

{% highlight ruby %}
fifa19.groupby(fifa19['Club'])['Nationality'].nunique().sort_values(ascending=False).head(11)
#untuk melihat pesepakbola dengan jumlah pemain dari negara berbeda paling sedikit, cukup ganti menjadi ascending=True
{% endhighlight %}

![26DIVa.jpg](https://iili.io/26DIVa.jpg)

![26DTiJ.jpg](https://iili.io/26DTiJ.jpg)

Terlihat bahwa Brighton & Hove Albion menjadi klub yang memiliki pemain dari negara yang berbeda yang terbanyak, yakni 21 negara. Sementara terdapat beberapa klub yang hanya memiliki pemain yang berasal dari 1 negara saja.

Setelah itu, kita akan mengubah feature `Real Face` dan `Preferred Foot` menjadi data numerikan dengan code berikut

{% highlight ruby %}
def face_to_num(data):
    if (data['Real Face'] == 'Yes'):
        return 1
    else:
        return 0
    
def preferred_foot(data):
    if (data['Preferred Foot'] == 'Right'):
        return 1
    else:
        return 0

fifa19['Real Face'] = fifa19.apply(face_to_num, axis=1)
fifa19['Preferred Foot'] = fifa19.apply(preferred_foot, axis=1)
{% endhighlight %}

### Data Splitting

Berikutnya kita akan split data menjadi data train dan data test untuk pembuatan model Machine Learning dan juga Deep Learning. Sebelum displit, kita akan mengambil terlebih dahulu feature-feature yang akan kita pakai untuk proses training data. Feature-feature yang akan kita ambil adalah feature-feature yang memiliki tipe data `int64` atau `float64`. Untuk codenya dapat dilihat dibawah ini.

{% highlight ruby %}
df = fifa19[['Age', 'Wage', 'Value', 'Special', 'Dribbling', 'Pace', 'Defending', 'Shooting', 'Passing', 'Physical', 'Potential', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Reactions', 'Composure', 'GKDiving', 'GKHandling', 'GKKicking',	'GKPositioning', 'GKReflexes', 'Overall']]
{% endhighlight %}

Setelah itu kita akan memisahkan feature dengan targetnya. Untuk feature kita beri nama X, sedangkan untuk target diberi nama y. Target dari dataset ini adalah `Overall`. Komposisi untuk data training dan data testnya adalah 80:20. Untuk split datanya kita memakai `train_test_split` dari library sckit-learn. Berikut merupakan codenya

{% highlight ruby %}
X = df.drop(['Overall'], axis=1)
y = df['Overall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
{% endhighlight %}

### Machine Learning Model

Berikutnya kita akan membuat model Machine Learning memakai 4 algoritma yaitu **Linear Regression**, **Support Vector Regressor**, **Random Forest Regressor**, dan **K-Nearest Neighbors Regressor**. Untuk proses trainingnya kita akan digunakan `Pipeline` dari sckit-learn.

**Linear Regression**

Pertama kita akan buat model pipelinenya terlebih dahulu yang terdiri dari scaler dan model machine learningnya. Scaler yang dipakai adalah `RobustScaler()`

{% highlight ruby %}
pipeline = Pipeline([('scaler', RobustScaler()), ('model', LinearRegression())])
pipeline.fit(X_train, y_train)
{% endhighlight %}

Kemudian akan dilakukan testing terhadap data train

{% highlight ruby %}
prediksi_linreg_train = pipeline.predict(X_train)
mse_train = mean_squared_error(y_train, prediksi_linreg_train)
r2_train = r2_score(y_train, prediksi_linreg_train)
{% endhighlight %}

dan berikut outputnya 

![26DnN1.jpg](https://iili.io/26DnN1.jpg)

Lalu, kita validasi terhadap data test

{% highlight ruby %}
prediksi_linreg = pipeline.predict(X_test)
mse = mean_squared_error(y_test, prediksi_linreg)
mae = mean_absolute_error(y_test, prediksi_linreg)
r2 = r2_score(y_test, prediksi_linreg)
{% endhighlight %}

dan berikut outputnya

![26DoDF.jpg](https://iili.io/26DoDF.jpg)

**Support Vector Regressor**

Pertama kita akan lakukan hyperparameter tuning terlebih dahulu memakai `RandomizedSearchCV()`. Parameter-parameter yang akan dituning adalah sebagai berikut

{% highlight ruby %}
C = [0.001, 0.01, 1.0, 10.0, 100.0, 1000.0]
gamma = [1, 0.1, 0.01, 0.001]

svm_grid = {'model__C' : [1.0, 10.0, 100.0, 1000.0],
            'model__gamma' : [1, 0.1, 0.01, 0.001]}
{% endhighlight %}

Kemudian kita akan buat model pipelinenya yang terdiri dari scaler dan model machine learningnya. Scaler yang dipakai adalah `RobustScaler()`

{% highlight ruby %}
pipeline_svm = Pipeline([('scaler', RobustScaler()), ('model', SVR(kernel='rbf'))])
{% endhighlight %}

Lalu kita akan tuning hyperparameternya

{% highlight ruby %}
svm_random = RandomizedSearchCV(pipeline_svm, svm_grid, cv=3, n_iter=50, verbose=2, random_state=10, n_jobs=-1)
svm_random.fit(X_train, y_train)
{% endhighlight %}

Setelah dituning, cek parameter terbaik hasil tuning

{% highlight ruby %}
svm_random.best_params_
{% endhighlight %}

dan diperoleh parameter terbaiknya adalah **C = 1000** dan **gamma = 0.01**. Lalu, parameter terbaik ini akan kita masukkan kedalam pipeline dan akan dilakukan training

{% highlight ruby %}
pipeline_svm = Pipeline([('scaler', RobustScaler()), ('model', SVR(kernel='rbf', C=1000, gamma=0.01))])
pipeline_svm.fit(X_train, y_train)
{% endhighlight %}

Setelah itu kita akan lakukan testing terhadap data train

{% highlight ruby %}
prediksi_svm_train = pipeline_svm.predict(X_train)
r2_train = r2_score(y_train, prediksi_svm_train)
mse_train = mean_squared_error(y_train, prediksi_svm_train)
print('r^2 score terhadap Data Train dengan Robust Scaler                       :', r2_train)
print('RMSE (Root Mean Squared Error) terhadap Data Train dengan Robust Scaler  :', np.sqrt(mse_train))
{% endhighlight %}

dan berikut outputnya

![26DRlR.jpg](https://iili.io/26DRlR.jpg)

Lalu, kita akan validasi dengan data testnya

{% highlight ruby %}
prediksi_svm = pipeline_svm.predict(X_test)
mse = mean_squared_error(y_test, prediksi_svm)
mae = mean_absolute_error(y_test, prediksi_svm)
r2 = r2_score(y_test, prediksi_svm)
print('MSE (Mean Squared Error) terhadap Data Test dengan Robust Scaler        :', mse)
print('MAE (Mean Absolute Error) terhadap Data Test dengan Robust Scaler       :', mae)
print('r^2 terhadap Data Test dengan Robust Scaler                             :', r2)
print('RMSE (Root Mean Squared Error) terhadap Data Test dengan Robust Scaler  :', np.sqrt(mse))
{% endhighlight %}

dan berikut outputnya

![26D5Sp.jpg](https://iili.io/26D5Sp.jpg)

**Random Forest Regressor**

Pertama kita akan lakukan hyperparameter tuning terlebih dahulu memakai `RandomizedSearchCV()`. Parameter-parameter yang akan dituning adalah sebagai berikut

{% highlight ruby %}
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'model__n_estimators' : n_estimators,
               'model__max_features' : max_features, 
               'model__max_depth' : max_depth,
               'model__min_samples_split' : min_samples_split,
               'model__min_samples_leaf' : min_samples_leaf,
               'model__bootstrap' : bootstrap}
{% endhighlight %}

Kemudian kita akan buat model pipelinenya yang terdiri dari scaler dan model machine learningnya. Scaler yang dipakai adalah `RobustScaler()`

{% highlight ruby %}
pipeline_rf = Pipeline([('scaler', RobustScaler()), ('model', RandomForestRegressor())])
{% endhighlight %}

Lalu kita akan tuning hyperparameternya

{% highlight ruby %}
rf_random = RandomizedSearchCV(pipeline_rf, random_grid, cv=3, n_iter=50, verbose=2, random_state=10, n_jobs=-1)
rf_random.fit(X_train, y_train)
{% endhighlight %}

Setelah dituning, cek parameter terbaik hasil tuning

{% highlight ruby %}
rf_random.best_params_
{% endhighlight %}

dan diperoleh parameter terbaiknya adalah **bootstrap = True** dan **max_depth=20**, **max_features='auto'**, **min_samples_leaf=1**, **min_samples_split=2**, dan **n_estimators=1800**. Lalu, parameter terbaik ini akan kita masukkan kedalam pipeline dan akan dilakukan training

{% highlight ruby %}
pipeline_rf = Pipeline([('scaler', RobustScaler()), ('model', RandomForestRegressor(bootstrap=True, max_depth=20, max_features='auto', min_samples_leaf=1, min_samples_split=2, n_estimators=1800))])
pipeline_rf.fit(X_train, y_train)
{% endhighlight %}

Setelah itu kita akan lakukan testing terhadap data train

{% highlight ruby %}
prediksi_rf_train = pipeline_rf.predict(X_train)
r2_train = r2_score(y_train, prediksi_rf_train)
mse_train = mean_squared_error(y_train, prediksi_rf_train)
print('r^2 score terhadap Data Train dengan Robust Scaler                       :', r2_train)
print('RMSE (Root Mean Squared Error) terhadap Data Train dengan Robust Scaler  :', np.sqrt(mse_train))
{% endhighlight %}

dan berikut outputnya

![26DYHN.jpg](https://iili.io/26DYHN.jpg)

Lalu, kita akan validasi dengan data testnya

{% highlight ruby %}
prediksi_rf = pipeline_rf.predict(X_test)
mse = mean_squared_error(y_test, prediksi_rf)
mae = mean_absolute_error(y_test, prediksi_rf)
r2 = r2_score(y_test, prediksi_rf)
print('MSE (Mean Squared Error) terhadap Data Test dengan Robust Scaler        :', mse)
print('MAE (Mean Absolute Error) terhadap Data Test dengan Robust Scaler       :', mae)
print('r^2 terhadap Data Test dengan Robust Scaler                             :', r2)
print('RMSE (Root Mean Squared Error) terhadap Data Test dengan Robust Scaler  :', np.sqrt(mse))
{% endhighlight %}

dan berikut outputnya

![26DaRI.jpg](https://iili.io/26DaRI.jpg)

**K-Nearest Neighbors Regressor**

Pertama kita akan lakukan hyperparameter tuning terlebih dahulu memakai `RandomizedSearchCV()`. Parameter-parameter yang akan dituning adalah sebagai berikut

{% highlight ruby %}
n_neighbors = [int(x) for x in np.linspace(start=1, stop=25, num=13)]

knn_grid = {'model__n_neighbors' : n_neighbors}
{% endhighlight %}

Kemudian kita akan buat model pipelinenya yang terdiri dari scaler dan model machine learningnya. Scaler yang dipakai adalah `RobustScaler()`

{% highlight ruby %}
pipeline_knn = Pipeline([('scaler', RobustScaler()), ('model', KNeighborsRegressor())])
{% endhighlight %}

Lalu kita akan tuning hyperparameternya

{% highlight ruby %}
knn_random = RandomizedSearchCV(pipeline_knn, knn_grid, cv=3, n_iter=50, verbose=2, random_state=10, n_jobs=-1)
knn_random.fit(X_train, y_train)
{% endhighlight %}

Setelah dituning, cek parameter terbaik hasil tuning

{% highlight ruby %}
knn_random.best_params_
{% endhighlight %}

dan diperoleh parameter terbaiknya adalah **n_neighbors=15**. Lalu, parameter terbaik ini akan kita masukkan kedalam pipeline dan akan dilakukan training

{% highlight ruby %}
pipeline_knn = Pipeline([('scaler', RobustScaler()), ('model', KNeighborsRegressor(n_neighbors=15))])
pipeline_knn.fit(X_train, y_train)
{% endhighlight %}

Setelah itu kita akan lakukan testing terhadap data train

{% highlight ruby %}
prediksi_knn_train = pipeline_knn.predict(X_train)
r2_train = r2_score(y_train, prediksi_knn_train)
mse_train = mean_squared_error(y_train, prediksi_knn_train)
print('r^2 score terhadap Data Train dengan Robust Scaler                       :', r2_train)
print('RMSE (Root Mean Squared Error) terhadap Data Train dengan Robust Scaler  :', np.sqrt(mse_train))
{% endhighlight %}

dan berikut outputnya

![26DcNt.jpg](https://iili.io/26DcNt.jpg)

Lalu, kita akan validasi dengan data testnya

{% highlight ruby %}
prediksi_knn = pipeline_knn.predict(X_test)
mse = mean_squared_error(y_test, prediksi_knn)
mae = mean_absolute_error(y_test, prediksi_knn)
r2 = r2_score(y_test, prediksi_knn)
print('MSE (Mean Squared Error) terhadap Data Test dengan Robust Scaler        :', mse)
print('MAE (Mean Absolute Error) terhadap Data Test dengan Robust Scaler       :', mae)
print('r^2 terhadap Data Test dengan Robust Scaler                             :', r2)
print('RMSE (Root Mean Squared Error) terhadap Data Test dengan Robust Scaler  :', np.sqrt(mse))
{% endhighlight %}

dan berikut outputnya

![26DlDX.jpg](https://iili.io/26DlDX.jpg)

### Deep Learning Model

Berikutnya kita akan coba membuat train data dengan model Deep Learning dengan multilayer perceptron. Libraries yang akan dipakai adalah sebagai berikut

{% highlight ruby %}
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import os
import datetime
{% endhighlight %}

Sebelum dilakukan training, data terlebih dahulu dipisahkan feature dan labelnya kemudian discale memakai `RobustScaler()`. Kemudian kita save model scalernya.

{% highlight ruby %}
X_dl = df.drop(['Overall'], axis=1)
y_dl = df['Overall']

rbst = RobustScaler()
rbst.fit(X_dl)
X_dl = rbst.transform(X_dl)
y_dl = rbst.fit_transform(df['Overall'].values.reshape(-1, 1)).flatten()

scalername = 'scalerobust.pkl' #Nama filenya
pickle.dump(rbst, open(scalername, 'wb')) 
{% endhighlight %}

Lalu, setelah itu feature dan labelnya dipisahkan menjadi data train dan data test dengan komposisi 80:20.

{% highlight ruby %}
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y_dl, test_size=0.2, random_state=10)
{% endhighlight %}

Setelah itu akan dibuat multilayer perceptron dengan arsitektur sebagai berikut

{% highlight ruby %}
model = Sequential()
model.add(Dense(21, input_dim=21, kernel_initializer='uniform', activation='relu')) #memakai 21 neuron dengan input_dim=21 karena ada 21 feature yang dipakai
model.add(Dense(10, kernel_initializer='uniform', activation='relu')) #deeper layer dengan 10 neuron
model.add(Dense(1, kernel_initializer='uniform')) #problem regression sehingga memakai 1 neuron, activation sigmoid

opt = SGD(learning_rate=0.001, momentum=0.9) #optimizer digunakan SGD

model.summary() #melihat summary dari model

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['CosineSimilarity']) #loss yang akan dipakai adalah mean squared error
{% endhighlight %}

Kemudian kita akan save modelnya dengan nama `weights_best_only.h5` dimana kita akan save model terbaiknya saja. Karena loss yang dipakai adalah mean squared error, kita akan save model dengan nilai `val_loss` yang paling kecil.

{% highlight ruby %}
filepath="weights_best_only.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) # Tempat dimana log tensorboard akan di
callbacks_list.append(TensorBoard(logdir, histogram_freq=1))
{% endhighlight %}

Lalu, kita akan train modelnya. 

{% highlight ruby %}
history = model.fit(X_train_dl, y_train_dl, batch_size=32, validation_data=(X_test_dl, y_test_dl), epochs=50, callbacks=callbacks_list, verbose=0)
{% endhighlight %}

Setelah itu kita akan lakukan validasi model terhadap data testing.

{% highlight ruby %}
predict_dl = model.predict(X_test_dl)
predict_dl = predict_dl.flatten()

mse = mean_squared_error(y_test_dl, predict_dl)
mae = mean_absolute_error(y_test_dl, predict_dl)
r2 = r2_score(y_test_dl, predict_dl)
print("MSE (Mean Squared Error)       :", mse)
print("MAE (Mean Absolute Error)      :", mae)
print("r^2 score                      :", r2)
print('RMSE (Root Mean Squared Error) :', np.sqrt(mse))
{% endhighlight %}

dan berikut outputnya

![2Zr9pt.jpg](https://iili.io/2Zr9pt.jpg)

Lalu, kita akan cek grafik `epoch_loss`dan grafik `epoch_cosine_similiarity` dengan Tensorboard

**Grafik Epoch-Loss**

![2Zrf44.jpg](https://iili.io/2Zrf44.jpg)

**Grafik Epoch-Cosine Similiarity**

![2ZrKGf.jpg](https://iili.io/2ZrKGf.jpg)

Berdasarkan grafik `epoch_loss`, terlihat bahwa model yang kita buat sudah bagus karena grafiknya sudah landai dan perbedaan nilai `loss` dan `val_loss`nya yang sangat kecil yang berarti hasil prediksi sudah sangat mendekati hasil ekspektasinya dengan *sweet spot*nya berada diepoch 47 dengan nilai `val_loss ` 0.011848. Hal ini pun diperkuat dengan nilai r2 scorenya yang mencapai 0.979 dan rmse nya yang sangat kecil yakni sebesar 0.1. Grafik `epoch_cosine_similiarity` pun menunjukkan nilai yang mendekati 1 yakni 0.9 dimana yang berarti nilai dari label test dan hasil prediksinya sudah mirip.

### Deployment Model


[sofifa]: https://sofifa.com/