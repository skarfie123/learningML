fi1eName=C\:/pics/test.PNG 
base64Image=$(base64 $fileName) 
jsonified="{\"image\":\"${base64Image}\"}" 
echo $jsonified >> data.json 
curl -X POST --data @data.json http://localhost:5000/predict 
rm data.json