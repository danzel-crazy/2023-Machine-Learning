
# device="cpu"
test_data = []
with open(f'/kaggle/input/hw5dataset/hw5/sample_submission.csv', newline='') as csvfile:
    for row in csv.reader(csvfile, delimiter=','):
        test_data.append(row)

test_ds1 = Task1Dataset(test_data, root=TEST_PATH, return_filename=True)
test_dl1 = DataLoader(test_ds1, batch_size=32, num_workers=4, drop_last=False, shuffle=False)
test_ds2 = Task2Dataset(test_data, root=TEST_PATH, return_filename=True)
test_dl2 = DataLoader(test_ds2, batch_size=32, num_workers=4, drop_last=False, shuffle=False)
test_ds3 = Task3Dataset(test_data, root=TEST_PATH, return_filename=True)
test_dl3 = DataLoader(test_ds3, batch_size=32, num_workers=4, drop_last=False, shuffle=False)


if os.path.exists('submission.csv'):
    file = open('submission.csv', 'w', newline='')
    csv_writer = csv.writer(file)
else:
    file = open('submission.csv', 'w', newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(["filename", "label"])
    
model1 = torchvision.models.resnet18(pretrained=True).to(device)
model1.fc = nn.Linear(in_features=512, out_features=10, bias=True).to(device)
model1.load_state_dict(torch.load("/kaggle/working/task1_model"))
# torch.load('/kaggle/working/task1_model_2', map_location=torch.device('cpu'))
model1.eval()

for image, filenames in test_dl1:
    image = image.to(device)
    
    pred = model1(image)
    pred = torch.argmax(pred, dim=1)
    
    for i in range(len(filenames)):
        print(str(pred[i].item()))
        csv_writer.writerow([filenames[i], str(pred[i].item())])
    del image, filenames

table = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
model2 = torchvision.models.resnet18(pretrained=True).to(device)
model2.fc = nn.Linear(in_features=512, out_features=72, bias=True).to(device)
model2.load_state_dict(torch.load("/kaggle/working/task2_model"))
# torch.load('/kaggle/working/task2_model_3', map_location=torch.device('cpu'))
model2.eval()
        
for image, filenames in test_dl2:
    image = image.to(device)
    pred = model2(image)
#     pred = torch.argmax(pred, dim=1)
    
    for i in range(len(filenames)):
        ans = ""
        large1_pred = torch.argmax(pred[i][0:36])
        large2_pred = torch.argmax(pred[i][36:72])
        ans = ans + table[large1_pred]
        ans = ans + table[large2_pred]
        print(ans)
        csv_writer.writerow([filenames[i], ans])
    del image, filenames

model3 = torchvision.models.resnet18(pretrained=True).to(device)
model3.fc = nn.Linear(in_features=512, out_features=144, bias=True).to(device)
model3.load_state_dict(torch.load("/kaggle/working/task3_model"))
# torch.load('/kaggle/working/task3_model_1', map_location=torch.device('cpu'))
model3.eval()

for image, filenames in test_dl3:
    image = image.to(device)
    
    pred = model3(image)
#     pred = torch.argmax(pred, dim=1)
    
    for i in range(len(filenames)):
        ans = ""
        large1_pred = torch.argmax(pred[i][0:36])
        large2_pred = torch.argmax(pred[i][36:72])
        large3_pred = torch.argmax(pred[i][72:108])
        large4_pred = torch.argmax(pred[i][108:144])
        ans = ans + table[large1_pred]
        ans = ans + table[large2_pred]
        ans = ans + table[large3_pred]
        ans = ans + table[large4_pred]
        print(ans)
        csv_writer.writerow([filenames[i], ans])
    del image, filenames

file.close()