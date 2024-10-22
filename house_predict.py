import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
video=f"""
   <style>
  .vid{{
      position: fixed;
	  right: 0;
	  bottom: 0;
      min-width: 100%; 
	  min-height: 100%;
    
  }}
   </style>
  
 <video autoplay loop muted class="vid">
 <source src="https://cdn.discordapp.com/attachments/1294905019388395563/1297964135580958801/7578544-hd_1280_720_30fps.mp4?ex=6717d6cc&is=6716854c&hm=a786b4c450fd04541f41c6dc100b399d6c5f80fb07f4a44be031fae4fad8d34f&">

 </video>

"""
# hot=1 if hot='yes' else 0

st.markdown(video,unsafe_allow_html=True)
st.header("House Price Prediction System")
col1,col2,col3=st.columns(3)
with col1:
    area=st.text_input("Area",placeholder="Enter area in squarefoot")
    bed=st.text_input("bedrooms",placeholder="Enter no of bedrooms ")
    bath=st.text_input("bathroom",placeholder="Enter no of bathroom ")
    story=st.text_input("stories",placeholder="Enter no of stories")
  
with col2:
    guest=st.text_input("guestroom",placeholder="type 0 for no and 1 for yes")
    base=st.text_input("basement",placeholder="type 0 for no and 1 for yes")
    hot=st.text_input("hotwaterheating",placeholder="Type 0 for no and 1 for yes ")
    air=st.text_input("airconditiomning",placeholder="Enter no of bedrooms ")
with col3:
    park=st.text_input("parking",placeholder="Enter 0 or 1")
    pref=st.text_input("prefarea",placeholder="prefarea")
    furnish=st.text_input("furnishingstatus",placeholder="Type 0/1/2")
    road=st.text_input("mainroad",placeholder="type 0 for no and 1 for yes")
#[area,bedrooms,stories,mainroad,guestroom,basement,hotwaterheating,¶
#airconditiomning,parking,,prefarea,furnishingstatus---independent--x]
li=[bed,bath,story,guest,base,hot,air,park,pref,furnish,road]
# st.write(li)


# Convert inputs to numeric (integer or float)
try:
    area = float(area)
    bed = int(bed)
    bath = int(bath)
    story = int(story)
    guest = int(guest)
    base = int(base)
    hot = int(hot)
    air = int(air)
    park = int(park)
    pref = int(pref)
    furnish = int(furnish)
    road = int(road)
except ValueError:
    st.error("Please make sure all inputs are valid numbers.")

print(li)
df=pd.read_csv('housing.csv')
encoder=LabelEncoder()
# main_road=encoder.fit_transform(df['mainroad'])
# df['mainroad']=main_road
# room=encoder.fit_transform(df['guestroom'])
# df['guestroom']=room
# base=encoder.fit_transform(df['basement'])
# df['basement']=base
# pre=encoder.fit_transform(df['prefarea'])
# df['prefarea']=pre
# fun=encoder.fit_transform(df['furnishingstatus'])
# df['furnishingstatus']=fun
# heat=encoder.fit_transform(df['hotwaterheating'])
# df['hotwaterheating']=heat
# air=encoder.fit_transform(df['airconditioning'])
# df['airconditioning']=air

df['mainroad'] = encoder.fit_transform(df['mainroad'])
df['guestroom'] = encoder.fit_transform(df['guestroom'])
df['basement'] = encoder.fit_transform(df['basement'])
df['prefarea'] = encoder.fit_transform(df['prefarea'])
df['furnishingstatus'] = encoder.fit_transform(df['furnishingstatus'])
df['hotwaterheating'] = encoder.fit_transform(df['hotwaterheating'])
df['airconditioning'] = encoder.fit_transform(df['airconditioning'])
scaler=StandardScaler()
scaler.fit(df[["area"]])


if st.button("View"):
    val=scaler.transform([[area]])
    st.write(val)
    li.append(val)
    model=joblib.load('house_price_1.pkl')
    answer=model.predict(li)
    st.subheader(f"Price of the House is - {answer[0]}")

