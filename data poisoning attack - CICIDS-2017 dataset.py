#Import scikit-learn dataset library
from sklearn import datasets
#Import svm model
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as metrics
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_curve, auc
import warnings

warnings.filterwarnings("ignore")
#Load dataset
botnet_detection_dataset = pd.read_csv('E:/study/label-flipping attack/Wednesday-workingHours.pcap_ISCX.csv')
botnet_detection_dataset.reset_index(drop=True, inplace=True)
botnet_detection_dataset.replace('', np.nan, inplace=True)
botnet_detection_dataset.fillna(0, inplace=True)
# botnet_detection_dataset.drop(columns=['state', 'proto', 'service', 'attack_cat'], inplace=True)
print(len(botnet_detection_dataset))
print(botnet_detection_dataset.head())
#split dataset into parts
def datasetSplitCompute(train_dataset, train_dataset_label,  num_parts):
    num_samples = len(train_dataset)
    print(num_samples)
    shuffledDataset = np.random.permutation(num_samples)
    part_size = num_samples / num_parts
    print(part_size)
    part_indices = []
    part_dataset = []
    partitioned_dataset = {}
    for num in range(0, num_parts):
        print(int((num)*part_size),":",int((num+1)*part_size))
        variable_dataset = "dataset_part"+str(num)
        variable_dataset_label = "dataset_part_label"+str(num)
        partitioned_dataset[variable_dataset] = train_dataset[int((num)*part_size):int((num+1)*part_size)]
        partitioned_dataset[variable_dataset_label] = train_dataset_label[int((num)*part_size):int((num+1)*part_size)]
    return partitioned_dataset


def randomSelect(dataset, label, poison):
    #taking 12.5% of dataset length
    label = label.values
    print(len(label))
    randomLength = int(len(dataset)*poison)
    #initiate 
    indexHolder = []
    random_integer = 0
    i = 0
    while(len(indexHolder)<randomLength):
        random_integer = random.randint(0, len(dataset)-1)
        if (random_integer not in indexHolder):
            # print("random integer: ", random_integer)
            # print(label[random_integer])
            if(label[random_integer] == 0):
                 label[random_integer] = 1
            else:
                label[random_integer] = 0
        indexHolder.append(random_integer)
        i+=1
    print("i: ", i)
    poisonedLabels = pd.DataFrame(label)
    poisonedLabels.rename(columns={'0': 'label'}, inplace=True)
    print(poisonedLabels.columns)
    return poisonedLabels

poisonPercentages = [0.10, 0.15, 0.25]

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(botnet_detection_dataset.drop(columns=["label"]),
                                                    #[['internet','access coarse location','access fine location','get tasks','change wifi state','write external storage','read phone state','system alert window','c2d message','camera','call phone','bluetooth','bluetooth admin', 'authenticate accounts', 'manage accounts', 'use credentials', 'use sip', 'write contacts', 'read contacts', 'read profile', 'write profile', 'read social stream', 'write social stream', 'write sms', 'read sms', 'receive sms', 'send sms', 'receive mms', 'receive wap push', 'record audio',  'read user dictionary', 'read cell broadcasts', 'read calendar', 'write calendar', 'read history bookmarks', 'write history bookmarks', 'read call log', 'write call log',  'process outgoing calls', 'nfc', 'change wimax state', 'change wifi multicasts state', 'change component enabled state', 'disable keyguard', 'battery stats', 'status bar',  'status bar service', 'clear app cache', 'access mock location', 'add voicemail', 'write media storage', 'write apn settings', 'read privileged phone state', 'call privileged',  'read frame buffer', 'set preferred applications', 'subscribed feeds write', 'inject events', 'mount unmount file systems', 'package usage stats', 'force stop packages',  'install packages', 'delete packages', 'broadcast package removed', 'access surface flinger', 'access cache filesystem', 'delete cache files', 'backup', 'set time', 'set orientation', 'interact across users full', 'internal system window', 'BIND INPUT METHOD', 'BIND WIDGET', 'bind appwidget ', 'bind wallpaper', 'bind accessibility service',  'bind directory search', 'clear app user data', 'read network usage history', 'update device stats', 'manage users', 'manage usb', 'account manager', 'device power',  'modify phone state', 'connectivity internal', 'master clear', 'reboot', 'mount format filesystems', 'stop app switches', 'HARDWARE TEST', 'FACTORY TEST', 'GLOBAL SEARCH CONTROL',  'send sms no confirmation', 'configure wifi display', 'allow any codec for playback', 'broadcast sms', 'broadcast wap push', 'BROADCAST PACKAGE ADDED', 'BROADCAST PACKAGE CHANGED',  'BROADCAST PACKAGE INSTALL', 'BROADCAST PACKAGE REPLACED', 'control location updates', 'ACCESS NETWORK STATE', 'ACCESS WIFI STATE', 'CHANGE NETWORK STATE', 'MODIFY AUDIO SETTINGS',  'FOREGROUND SERVICE', 'GET ACCOUNTS', 'READ EXTERNAL STORAGE', 'RECEIVE BOOT COMPLETED', 'VIBRATE', 'WAKE LOCK', 'BILLING RECEIVE', 'BIND GET INSTALL REFERRER SERVICE', 'READ SETTINGS', 'WRITE SETTINGS', 'READ G SERVICES', 'ACTIITY RECOGNITION', 'USE FINGERPRINT', 'USE BIOMETRIC', 'SET ALARM', 'READ SYNC SETTINGS', 'WRITE SYNC SETTINGS',  'READ SYNC STATS', 'SYSTEM OVERLAY WINDOW', 'CHECK LICENSE', 'BROADCAST STICKY', 'EXPAND STATUS BAR', 'FLASHLIGHT', 'GET PACKAGE SIZE', 'KILL BACKGROUND PROCESSES',  'RECORDER TASKS', 'INSTALL SHORTCUT', 'UNINSTALL SHORTCUT', 'REQUEST IGNORE BATTERY OPTIMIZATIONS ', 'SET WALLPAPER', 'SET WALLPAPER HINTS', 'REQUEST DELETE PACKAGES', 'REQUEST INSTALL PACKAGES', 'RESTART PACKAGES', 'DOWNLOAD WITHOUT NOTIFICATION', 'MAPS RECEIVE', 'RECEIVE USER PRESENT', 'RECEIVE ADM MESSAGE', 'READ INTERNAL STORAGE', 'WRITE INTERNAL STORAGE', 'WRITE USER DICTIONARY', 'UPDATE SHORTCUT', 'UPDATE COUNT', 'READ APP BADGE', 'CHANGE BADGE', 'UPDATE BADGE', 'READ', 'WRITE', 'BROADCAST BADGE',  'PROVIDER INSERT BADGE', 'BADGE COUNT READ', 'BADGE COUNT WRITE', 'UPDATE APP BADGE', 'YAHOO INTER APP', 'ACCESS', 'ACCESS GPS', 'ACCESS ASSISTED GPS', 'ACCESS LOCATION',  'ACCESS PERMISSION', 'ACCESS PUSHAGENT', 'ACCESS COARSE UPDATES', 'ACCESS PROVIDER', 'ACCESS CONTENT PROVIDER', 'ACCESS LOCATION EXTRA COMMANDS', 'ACCESS SUPERUSER', 'ACCESS NOTIFICATION POLICY', 'ACCESS ALL DOWNLOADS', 'ACCESS BLUETOOTH SHARE', 'ACCESS DOWNLOAD MANAGER', 'ACCESS DOWNLOAD MANAGER ADVANCED', 'ACCESS FINGERPRINT MANAGER', 'ACCESS DATA', 'ACCESS USER DATA', 'ACCESS SHARED DATA', 'ACCESS LAUNCHER DATA', 'ACCESS PHOTO LAB', 'ACCESS CHECKIN', 'ACCESS MOTOROLA PRIVACY SECONDARY',  'ACCESS THRID PARTY APP AUTHORIZATION', 'ACCESS COURSE LOCATION', 'ACCESS BACKGROUND LOCATION', 'ACCESS BACKGROUND SERVICE', 'ACCESS BACKSCREEN', 'ACCESS FIND LOCATION',  'ACCESS LUCY', 'ACCESS SDK', 'ACCESS SOCIALNETWORK SERVICE', 'ACCESS WEATHERCLOCK PROVIDER', 'PERMISSION', 'PERMISSION SUGARSYNC SERVICE', 'OBSERVE GRANT REVOKE PERMISSIONS', 'GRANT RUNTIME PERMISSIONS', 'CLOUD PERMISSION', 'SERVICE PERMISSION', 'PERMISSION MANAGE OVERLAY', 'PERMISSION STOP WATCHDOG AFTER SCREEN OFF', 'ACTION MANAGE OVERLAY PERMISSION',  'CLIPBOARDAVESERVICE PERMISSION', 'PERMISSION SAFE BROADCAST', 'PERMISSION SEPARATED PROCESS', 'READ PERMISSION', 'WRITE PERMISSION', 'IDS PERMISSION', 'CACHE PERMISSION',  'TEMPORARY DISABLE', 'FRAMEWORK SECURITY', 'GOLDENEYE SECURITY', 'MANAGE DOCUMENTS', 'MANAGE OWN CALLS', 'ANSWER PHONE CALLS', 'MANAGE SERVICE', 'FB APP COMMUNICATION', 'AM COMMUNICATION', 'READ CREDENTIALS', 'READ WRITE PROVIDER', 'JPUSH MESSAGE', 'SET TIME ZONE', 'UPDATE CONFIG', 'READ DATABASE', 'WRITE DATABASE', 'APP DEFAULT', 'APP PLATFORM',  'APP MEDIA', 'APP HSP', 'WATCH APP TYPE', 'ACCESSORY FRAMEWORK', 'ENABLE NOTIFICATION', 'WRITE USE APP FEATURE SURVEY', 'CROSS PROCESS BROADCAST MANAGER', 'BODY SENSORS',  'SENSOR ENABLE', 'SENSOR INFO', 'READ EPG DATA', 'WRITE EPG DATA', 'READ CARD DATA', 'WRITE CARD DATA', 'READ OWNER DATA', 'WRITE OWNER DATA', 'READ DATA', 'UA DATA',  'BIND DATA CONSUMER', 'READ EXTENSION DATA', 'BIND NOTIFICATION LISTENER SERVICE', 'BIND JOB SERVICE', 'INTERNAL BOADCAST', 'SECURED BROADCAST', 'DUID READ PROVIDER',  'PROVIDE BACKGROUND', 'GOOGLE PHOTOS', 'GOOGLE AUTH', 'CAPTURE AUDIO OUTPUT', 'CAPTURE AUDIO HOTWORD', 'CAPTURE VIDEO OUTPUT', 'CAPTURE SECURE VIDEO OUTPUT', 'GET STATE', 'DRAW CROP', 'EDIT', 'BIND', 'TOKEN', 'PREFETCH', 'LOCATION', 'INFO', 'KEYSTRING', 'PROFILER', 'MESSAGE', 'BADGING', 'LOAD', 'SAVE', 'AUTHENTICATE', 'RECEIVER', 'SEND', 'STORAGE',  'THEME', 'ASSISTANT', 'QSB', 'RESTRICTED', 'REGISTER', 'SMARTCARD', 'WEBLINK', 'AAM2', 'LIBRARY', 'CONFIG', 'PLUGIN', 'STATUS', 'UBLS WRITE LOG', 'SECURE VIEW', 'HMS ZERO',  'MEDIAMANAGER ACCESS MM', 'GENERAL', 'USE SOCIAL COMPONENT', 'USE PLUGINSERVICE', 'USE SOCIAL SERVICE', 'FEATUREDB ACCESS', 'RECEIVE MCS MESSAGE', 'MIPUSH RECEIVE', 'AUTH SERVICE', 'SYSTEM UI VISIBILITY EXTENSION', 'WRITE PUSHINFOPROVIDER', 'PROVIDER ACCESS MODIFY CONFIGURATION', 'READ CAMERA STATE', 'CAMERA ADDON', 'TIGER BLAST ACTION',  'INTERACT BLUR SERVICE', 'READ BRSETTINGS', 'WRITE BRSETTINGS', 'READ CONTENT PROVIDER', 'READ PROVIDER', 'SET NAVBAR BACKGROUNDCOLOR ', 'ACTION NEW ITEMS PROCESSED',  'MEDIA CONTENT CONTROL', 'LAUNCH PERSONAL PAGE SERVICE', 'PERSONAL MEDIA', 'CAR SPEED', 'A4S SEND', 'EXPORT SNCF', 'UPDATE STICKER INDEX', 'READ PHONE NUMBERS', 'APP INSTALL API',  'USES POLICY FORCE LOCK', 'CONTROL LIGHT', 'UPDATE APP OPS STATS', 'COARSE LOCATION', 'FINE LOCATION', 'READ BUDYY', 'WRITE BUDDY', 'WRITE ACCOUNT', 'READ ACCOUNT',  'READ HIACCOUNT PROVIDER', 'READ XACCOUNT PROVIDER', 'WRITE XACCOUNT PROVIDER', 'QUICKBOOT POWERON', 'SMS RECEIVED', 'UNPROTECTED API ACCESS', 'RECORD VIDEO', 'LOCATION HARDWARE',  'BROADCAST MESSAGE', 'BROADCAST RECEIVER', 'NEW MESSAGE', 'PUSHIO MESSAGE', 'NEW OUTGOING CALL', 'RECEIVE FEED', 'GET CLIPS', 'READ CLIPS', 'WRITE CLIPS', 'SENIOR ENABLE',  'SENIOR INFO', 'RECEIVE SCREEN OFF', 'RECEIVE SCREEN ON', 'OPPO COMPONENT SAFE', 'CLOUD MANAGER', 'ANDROID ID', 'SEND DOWNLOAD COMPLETED INTENTS', 'PIC SELECT',  'PAY THROUGH BAZAAR', 'PUSH SERVICE', 'RESANA ADS', 'READ ATTACHEMENT', 'READ GMAIL', 'WRITE GMAIL', 'ANT', 'ANT ADMIN', 'READ SOCIAL DATABASE', 'WRITE SOCIAL DATABASE',  'LAYER PUSH', 'NETWORK ACCESS', 'NETWORK', 'DEVICE STATS', 'RAISED THREAD PRIORITY', 'CONTENT READ', 'PUBLISH CUSTOM FILE', 'READ DEEPLINK DATABASE', 'SUBSCRIBED FEEDS READ',  'SHARE DATA', 'DATA RESET', 'LUCY BR Status Update', 'ADD TO DO IT LATER', 'SECURITY ACTIVITY', 'HW SIGNATURE OR SYSTEM', 'APPLY THEME', 'CALL DEMOAPP', 'FLAG SHOW WHEN LOCKED',  'REAL GET TASKS', 'GETTASKS', 'APPLOCK DB ACCESS', 'LOCAL MAC ADDRESS', 'READ SOCIALNETWORK DATA', 'WRITE SOCIALNETWORK DATA', 'EXTERNAL DATA', 'SHOW CUSTOM CONTENT',  'PROTECTED DEEPLINKING', 'DIRECT APP THREAD STORE SERVICE', 'USE ALML', 'BAIDU LOCATION SERVICE', 'TYPE APPLICATION OVERLAY', 'SYNC USER', 'C2D MESSAGEfcqq', 'READ SECURE SETTINGS',  'PERSISTENT ACTIVITY', 'INSTALL DRM', 'RECEIVE FIRST LOAD BROADCAST', 'RECEIVE LAUNCH BROADCASTS', 'MANAGE VOICE KEYPHRASES', 'READ STATE', 'READ PREFERENCES', 'BOOT COMPLETED',  'READ WALLPAPER INTERNAL', 'NOTIFY ROTATING WALLPAPER CHANGED', 'RAW STM401', 'WRITE PHONE STATE', 'PHONE STATE', 'GET BRANDING', 'ADVANCED APPMODE', 'SCROBBLET PRIVACY SERVICE',  'SYSTEMPROPERTIES', 'MY PACKAGE REPLACED', 'INTERAPP', 'HANDLE PUSH', 'HANDLE UNSAFE APK', 'MFC ACCESS', 'INTENTS', 'EBAY USER CONFIG', 'GALLERY PROVIDER', 'PER ACCOUNT TYPE',  'CONTENTBLOCKER', 'TILES ACCESS', 'NETWORK PROVIDER', 'PROCESS OUTGOING SMS', 'ADD SYSTEM SERVICE', 'KEYGUARD', 'FULLSCREEN.FULL', 'NFC SE', 'NFC TRANSACTION', 'TRANSACTION EVENT', 'GEAR COMMUNICATION', 'SCOTIBANK BLPM', 'SEND RESPOND VIA MESSAGE ', 'RECEIVE SENDTO', 'WRITE SECURE', 'ADD SYSTEM SERVICE', 'PROCESS INCOMING CALLS', 'USER PRESENT',  'PERMISSION NAME', 'EXPORT', 'READ LOGS']], 
                                                    botnet_detection_dataset[["label"]], 
                                                    test_size=0.3,random_state=109) # 70% training and 30% test

#Create a svm Classifier
# clf = svm.SVC(kernel='linear') # Linear Kernel
# SGDClf = SGDClassifier(loss="hinge", penalty="l2", max_iter=17)
# # LRClf = LinearRegression()
# RFClf = RandomForestClassifier()
# GNBClf = GaussianNB()
KNNClf = KNeighborsClassifier(n_neighbors=3)
# DTClf = tree.DecisionTreeClassifier()
# # LogRegClf = LogisticRegression()
# PerceptronClf = Perceptron()
# MLPClf = MLPClassifier()

#Train the model using the training sets
#print("X_train: ", botnet_detection_dataset[botnet_detection_dataset.isna().any(axis=1)])
# clf.fit(X_train, y_train)
# SGDClf.fit(X_train, y_train)
# # LRClf.fit(X_train, y_train)
# RFClf.fit(X_train, y_train)
# GNBClf.fit(X_train, y_train)
KNNClf.fit(X_train, y_train)
# DTClf.fit(X_train, y_train)
# LogRegClf.fit(X_train, y_train)
# PerceptronClf.fit(X_train, y_train)
# MLPClf.fit(X_train, y_train)

#Predict the response for test dataset
# y_pred = clf.predict(X_test)
# y_pred_sgd = SGDClf.predict(X_test)
# # y_pred_lr = LRClf.predict(X_test)
# y_pred_rf = RFClf.predict(X_test)
# y_pred_gnb = GNBClf.predict(X_test)
y_pred_knn = KNNClf.predict(X_test)
# y_pred_dt = DTClf.predict(X_test)
# # y_pred_logReg = LogRegClf.predict(X_test)
# y_pred_perceptron = PerceptronClf.predict(X_test)
# y_pred_mlp = MLPClf.predict(X_test)



# accuracy_clean_model_svm = accuracy_score(y_test, y_pred)
# print("clean model accuracy: ", accuracy_clean_model_svm)

# accuracy_clean_model_sgd = accuracy_score(y_test, y_pred_sgd)*100
# precision_clean_model_sgd = precision_score(y_test, y_pred_sgd)
# f1_clean_model_sgd = f1_score(y_test, y_pred_sgd)
# recall_clean_score_sgd = recall_score(y_test, y_pred_sgd)
# fpr_clean_model_sgd, tpr_clean_model_sgd, threshold_clean_model_sgd = roc_curve(y_test, y_pred_sgd)
# print("clean model accuracy SGD: ", accuracy_clean_model_sgd, ", precision: ", precision_clean_model_sgd, ", f1_score: ", f1_clean_model_sgd, ", recall: ", recall_clean_score_sgd, ", fpr: ", fpr_clean_model_sgd)

# accuracy_clean_model_lr = accuracy_score(y_test, y_pred_lr)*100
# precision_clean_model_lr = precision_score(y_test, y_pred_lr)
# f1_clean_model_lr = f1_score(y_test, y_pred_lr)
# recall_clean_score_lr = recall_score(y_test, y_pred_lr)
# fpr_clean_model_lr, tpr_clean_model_lr, threshold_clean_model_lr = roc_curve(y_test, y_pred_lr)
# print("clean model accuracy LR: ", accuracy_clean_model_lr, ", precision: ", precision_clean_model_lr, ", f1_score: ", f1_clean_model_lr, ", recall: ", recall_clean_score_lr, ", fpr: ", fpr_clean_model_lr)

# accuracy_clean_model_rf = accuracy_score(y_test, y_pred_rf)*100
# precision_clean_model_rf = precision_score(y_test, y_pred_rf)
# f1_clean_model_rf = f1_score(y_test, y_pred_rf)
# recall_clean_score_rf = recall_score(y_test, y_pred_rf)
# fpr_clean_model_rf, tpr_clean_model_rf, threshold_clean_model_rf = roc_curve(y_test, y_pred_rf)
# print("clean model accuracy RF: ", accuracy_clean_model_rf, ", precision: ", precision_clean_model_rf, ", f1_score: ", f1_clean_model_rf, ", recall: ", recall_clean_score_rf, ", fpr: ", fpr_clean_model_rf)

# accuracy_clean_model_gnb = accuracy_score(y_test, y_pred_gnb)*100
# precision_clean_model_gnb = precision_score(y_test, y_pred_gnb)
# f1_clean_model_gnb = f1_score(y_test, y_pred_gnb)
# recall_clean_score_gnb = recall_score(y_test, y_pred_gnb)
# fpr_clean_model_gnb, tpr_clean_model_gnb, threshold_clean_model_gnb = roc_curve(y_test, y_pred_gnb)
# print("clean model accuracy Gaussian Naive Bayes: ", accuracy_clean_model_gnb, ", precision: ", precision_clean_model_gnb, ", f1_score: ", f1_clean_model_gnb, ", recall: ", recall_clean_score_gnb, ", fpr: ", fpr_clean_model_gnb)

accuracy_clean_model_knn = accuracy_score(y_test, y_pred_knn)*100
# precision_clean_model_knn = precision_score(y_test, y_pred_knn)
# f1_clean_model_knn = f1_score(y_test, y_pred_knn)
# recall_clean_score_knn = recall_score(y_test, y_pred_knn)
# fpr_clean_model_knn, tpr_clean_model_knn, threshold_clean_model_knn = roc_curve(y_test, y_pred_knn)
print("clean model accuracy KNN: ", accuracy_clean_model_knn)
# , ", precision: ", precision_clean_model_knn, ", f1_score: ", f1_clean_model_knn, ", recall: ", recall_clean_score_knn, ", fpr: ", fpr_clean_model_knn)

# accuracy_clean_model_dt = accuracy_score(y_test, y_pred_dt)*100
# precision_clean_model_dt = precision_score(y_test, y_pred_dt)
# f1_clean_model_dt = f1_score(y_test, y_pred_dt)
# recall_clean_score_dt = recall_score(y_test, y_pred_dt)
# fpr_clean_model_dt, tpr_clean_model_dt, threshold_clean_model_dt = roc_curve(y_test, y_pred_dt)
# print("clean model accuracy DT: ", accuracy_clean_model_dt, ", precision: ", precision_clean_model_dt, ", f1_score: ", f1_clean_model_dt, ", recall: ", recall_clean_score_dt, ", fpr: ", fpr_clean_model_dt)

# accuracy_clean_model_logreg = accuracy_score(y_test, y_pred_logReg)*100
# precision_clean_model_logreg = precision_score(y_test, y_pred_logReg)
# f1_clean_model_logreg = f1_score(y_test, y_pred_logReg)
# recall_clean_score_logreg = recall_score(y_test, y_pred_logReg)
# fpr_clean_model_logreg, tpr_clean_model_logreg, threshold_clean_model_logreg = roc_curve(y_test, y_pred_logReg)
# print("clean model accuracy Logistic Regression: ", accuracy_clean_model_logreg, ", precision: ", precision_clean_model_logreg, ", f1_score: ", f1_clean_model_logreg, ", recall: ", recall_clean_score_logreg, ", fpr: ", fpr_clean_model_logreg)

# accuracy_clean_model_perceptron = accuracy_score(y_test, y_pred_perceptron)*100
# precision_clean_model_perceptron = precision_score(y_test, y_pred_perceptron)
# f1_clean_model_perceptron = f1_score(y_test, y_pred_perceptron)
# recall_clean_score_perceptron = recall_score(y_test, y_pred_perceptron)
# fpr_clean_model_perceptron, tpr_clean_model_perceptron, threshold_clean_model_perceptron = roc_curve(y_test, y_pred_perceptron)
# print("clean model accuracy Perceptron: ", accuracy_clean_model_perceptron, ", precision: ", precision_clean_model_perceptron, ", f1_score: ", f1_clean_model_perceptron, ", recall: ", recall_clean_score_perceptron, ", fpr: ", fpr_clean_model_perceptron)

# accuracy_clean_model_mlp = accuracy_score(y_test, y_pred_mlp)*100
# precision_clean_model_mlp = precision_score(y_test, y_pred_mlp)
# f1_clean_model_mlp = f1_score(y_test, y_pred_mlp)
# recall_clean_score_mlp = recall_score(y_test, y_pred_mlp)
# fpr_clean_model_mlp, tpr_clean_model_mlp, threshold_clean_model_mlp = roc_curve(y_test, y_pred_mlp)
# print("clean model accuracy MLP: ", accuracy_clean_model_mlp, ", precision: ", precision_clean_model_mlp, ", f1_score: ", f1_clean_model_mlp, ", recall: ", recall_clean_score_mlp, ", fpr: ", fpr_clean_model_mlp)
#create function to manipulate dataset label and draw poisoning attack
print("concatenated dataset: ", len(X_train), len(y_train))
# dataset = datasetSplitCompute(X_train, y_train, 4)

for poison in poisonPercentages:
    y_train = randomSelect(X_train, y_train, poison)
    #Create a svm Classifier
    # RandomLabelPoisonModel = svm.SVC(kernel='linear') # Linear Kernel
    #Train the model using the training sets
    # RandomLabelPoisonModel.fit(X_train, y_train)
    # SGDClf.fit(X_train, y_train)
    # # LRClf.fit(X_train, y_train)
    # RFClf.fit(X_train, y_train)
    # GNBClf.fit(X_train, y_train)
    KNNClf.fit(X_train, y_train)
    # DTClf.fit(X_train, y_train)
    # # LogRegClf.fit(X_train, y_train)
    # PerceptronClf.fit(X_train, y_train)
    # MLPClf.fit(X_train, y_train)

    #Predict the response for test dataset
    # y_poison_pred = RandomLabelPoisonModel.predict(X_test)
    # accuracy_poisoned_model = accuracy_score(y_test, y_poison_pred)

    # print("poisoned response: ", y_poison_pred)
    # print("accuracy of poisoned model: ", accuracy_poisoned_model)

    # y_poison_pred_sgd = SGDClf.predict(X_test)
    # accuracy_poisoned_model_sgd = accuracy_score(y_test, y_poison_pred_sgd)*100
    # precision_poisoned_model_sgd = precision_score(y_test, y_poison_pred_sgd)
    # f1_poisoned_model_sgd = f1_score(y_test, y_poison_pred_sgd)
    # recall_poisoned_model_sgd = recall_score(y_test, y_poison_pred_sgd)
    # fpr_poisoned_model_sgd, tpr_poisoned_model_sgd, threshold_poisoned_model_sgd = roc_curve(y_test, y_poison_pred_sgd)

    # print("accuracy of poisoned model SGD: ", accuracy_poisoned_model_sgd, ", precision: ", precision_poisoned_model_sgd, ", f1_score: ", f1_poisoned_model_sgd, ", recall: ", recall_poisoned_model_sgd, ", fpr: ", fpr_poisoned_model_sgd)

    # y_poison_pred_lr = LRClf.predict(X_test)
    # accuracy_poisoned_model_lr = metrics.r2_score(y_test, y_poison_pred_lr)
    # precision_poisoned_model_lr = precision_score(y_test, y_poison_pred_lr)
    # f1_poisoned_model_lr = f1_score(y_test, y_poison_pred_lr)
    # recall_poisoned_model_lr = recall_score(y_test, y_poison_pred_lr)
    # fpr_poisoned_model_lr, tpr_poisoned_model_lr, threshold_poisoned_model_lr = roc_curve(y_test, y_poison_pred_lr)

    # print("accuracy of poisoned model LR: ", accuracy_poisoned_model_lr, ", precision: ", precision_poisoned_model_lr, ", f1_score: ", f1_poisoned_model_lr, ", recall: ", recall_poisoned_model_lr, ", fpr: ", fpr_poisoned_model_lr)

    # y_poison_pred_rf = RFClf.predict(X_test)
    # accuracy_poisoned_model_rf = accuracy_score(y_test, y_poison_pred_rf)*100
    # precision_poisoned_model_rf = precision_score(y_test, y_poison_pred_rf)
    # f1_poisoned_model_rf = f1_score(y_test, y_poison_pred_rf)
    # recall_poisoned_model_rf = recall_score(y_test, y_poison_pred_rf)
    # fpr_poisoned_model_rf, tpr_poisoned_model_rf, threshold_poisoned_model_rf = roc_curve(y_test, y_poison_pred_rf)

    # print("accuracy of poisoned model RF: ", accuracy_poisoned_model_rf, ", precision: ", precision_poisoned_model_rf, ", f1_score: ", f1_poisoned_model_rf, ", recall: ", recall_poisoned_model_rf, ", fpr: ", fpr_poisoned_model_rf)

    # y_poison_pred_gnb = GNBClf.predict(X_test)
    # accuracy_poisoned_model_gnb = accuracy_score(y_test, y_poison_pred_gnb)*100
    # precision_poisoned_model_gnb = precision_score(y_test, y_poison_pred_gnb)
    # f1_poisoned_model_gnb = f1_score(y_test, y_poison_pred_gnb)
    # recall_poisoned_model_gnb = recall_score(y_test, y_poison_pred_gnb)
    # fpr_poisoned_model_gnb, tpr_poisoned_model_gnb, threshold_poisoned_model_gnb = roc_curve(y_test, y_poison_pred_gnb)

    # print("accuracy of poisoned model Gaussian Naive Bayes: ", accuracy_poisoned_model_gnb, ", precision: ", precision_poisoned_model_gnb, ", f1_score: ", f1_poisoned_model_gnb, ", recall: ", recall_poisoned_model_gnb, ", fpr: ", fpr_poisoned_model_gnb)

    y_poison_pred_knn = KNNClf.predict(X_test)
    accuracy_poisoned_model_knn = accuracy_score(y_test, y_poison_pred_knn)*100
    # precision_poisoned_model_knn = precision_score(y_test, y_poison_pred_knn)
    # f1_poisoned_model_knn = f1_score(y_test, y_poison_pred_knn)
    # recall_poisoned_model_knn = recall_score(y_test, y_poison_pred_knn)
    # fpr_poisoned_model_knn, tpr_poisoned_model_knn, threshold_poisoned_model_knn = roc_curve(y_test, y_poison_pred_knn)

    print("accuracy of poisoned model KNN: ", accuracy_poisoned_model_knn)
    # , ",\n precision: ", precision_poisoned_model_knn, ",\n f1_score: ", f1_poisoned_model_knn, ",\n recall: ", recall_poisoned_model_knn, ",\n fpr: ", fpr_poisoned_model_knn)

    # y_poison_pred_dt = DTClf.predict(X_test)
    # accuracy_poisoned_model_dt = accuracy_score(y_test, y_poison_pred_dt)*100
    # precision_poisoned_model_dt = precision_score(y_test, y_poison_pred_dt)
    # f1_poisoned_model_dt = f1_score(y_test, y_poison_pred_dt)
    # recall_poisoned_model_dt = recall_score(y_test, y_poison_pred_dt)
    # fpr_poisoned_model_dt, tpr_poisoned_model_dt, threshold_poisoned_model_dt = roc_curve(y_test, y_poison_pred_dt)

    # print("accuracy of poisoned model DT: ", accuracy_poisoned_model_dt, ",\n precision: ", precision_poisoned_model_dt, ",\n f1_score: ", f1_poisoned_model_dt, ",\n recall: ", recall_poisoned_model_dt, ",\n fpr: ", fpr_poisoned_model_dt)

    # # y_poison_pred_logreg = LogRegClf.predict(X_test)
    # # accuracy_poisoned_model_logreg = accuracy_score(y_test, y_poison_pred_logreg)*100
    # # precision_poisoned_model_logreg = precision_score(y_test, y_poison_pred_logreg)
    # # f1_poisoned_model_logreg = f1_score(y_test, y_poison_pred_logreg)
    # # recall_poisoned_model_logreg = recall_score(y_test, y_poison_pred_logreg)
    # # fpr_poisoned_model_logreg, tpr_poisoned_model_logreg, threshold_poisoned_model_logreg = roc_curve(y_test, y_poison_pred_logreg)

    # # print("accuracy of poisoned model Logistic Regression: ", accuracy_poisoned_model_logreg, ", precision: ", precision_poisoned_model_logreg, ", f1_score: ", f1_poisoned_model_logreg, ", recall: ", recall_poisoned_model_logreg, ", fpr: ", fpr_poisoned_model_logreg)

    # y_poison_pred_perceptron = PerceptronClf.predict(X_test)
    # accuracy_poisoned_model_perceptron = accuracy_score(y_test, y_poison_pred_perceptron)*100
    # precision_poisoned_model_perceptron = precision_score(y_test, y_poison_pred_perceptron)
    # f1_poisoned_model_perceptron = f1_score(y_test, y_poison_pred_perceptron)
    # recall_poisoned_model_perceptron = recall_score(y_test, y_poison_pred_perceptron)
    # fpr_poisoned_model_perceptron, tpr_poisoned_model_perceptron, threshold_poisoned_model_perceptron = roc_curve(y_test, y_poison_pred_perceptron)

    # print("accuracy of poisoned model Perceptron: ", accuracy_poisoned_model_perceptron, ", precision: ", precision_poisoned_model_perceptron, ", f1_score: ", f1_poisoned_model_perceptron, ", recall: ", recall_poisoned_model_perceptron, ", fpr: ", fpr_poisoned_model_perceptron)

    # y_poison_pred_mlp = MLPClf.predict(X_test)
    # accuracy_poisoned_model_mlp = accuracy_score(y_test, y_poison_pred_mlp)*100
    # precision_poisoned_model_mlp = precision_score(y_test, y_poison_pred_mlp)
    # f1_poisoned_model_mlp = f1_score(y_test, y_poison_pred_mlp)
    # recall_poisoned_model_mlp = recall_score(y_test, y_poison_pred_mlp)
    # fpr_poisoned_model_mlp, tpr_poisoned_model_mlp, threshold_poisoned_model_mlp = roc_curve(y_test, y_poison_pred_mlp)

    # print("accuracy of poisoned model MLP: ", accuracy_poisoned_model_mlp, ", precision: ", precision_poisoned_model_mlp, ", f1_score: ", f1_poisoned_model_mlp, ", recall: ", recall_poisoned_model_mlp, ", fpr: ", fpr_poisoned_model_mlp)