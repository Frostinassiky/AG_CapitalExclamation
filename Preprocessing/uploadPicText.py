import cognitive_face as CF
import time
KEY1 = '01efa897d01a4e2db732123865d6a504'  # Replace with a valid subscription key (keeping the quotes in place).
KEY2 = '7de5c0e0cf62438a9cd52a6b0da5a90e'
KEY = [KEY1,KEY2]
CF.Key.set(KEY)
amout = 5338
genderRes = {'0':'test'}
for k in range(amout):
    img_path = '/home/xum/Documents/WIDER FACE/collect/'+str(k+1)+'.png'
    #CF.Key.set(KEY[k%2])
    CF.Key.set(KEY[0])
    # upload pic
    flag = 1
    while flag:
        try:
            result = CF.face.detect(img_path, attributes='gender')
            flag = 0
        except:
            print "Rate limit is exceeded. Try again later."
            time.sleep(5)
    # read and save result
    try:
        gender = result[0]['faceAttributes']['gender']
        print str(k + 1) + ': ' + gender
        genderRes[str(k + 1)] = gender
    except:
        print str(k + 1) + '('+'can not find face): ', result
    time.sleep(1)

print genderRes
