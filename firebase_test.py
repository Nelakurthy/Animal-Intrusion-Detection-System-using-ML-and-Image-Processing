import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Fetch the service account key JSON file contents
cred = credentials.Certificate('key.json')

# Initialize the app with a service account, granting admin privileges

firebase_admin.initialize_app(cred, {
    'databaseURL': "https://homeautomation-fbfc2-default-rtdb.firebaseio.com/"
})

ref = db.reference('TRAFFIC')
ref.set({'SIGN': "0"})



print(ref.get())


