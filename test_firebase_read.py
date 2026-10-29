import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://soil-monitoring-system-db-default-rtdb.firebaseio.com"
})

print(db.reference("latest/device01").get())

history = db.reference("readings/device01") \
    .order_by_child("timestamp").limit_to_last(100).get()
print(len(history), "readings")