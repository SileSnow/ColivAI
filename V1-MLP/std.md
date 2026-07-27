# Data ingestion  
***  
# Data preprocessing  

1. define transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

2.create Datasets with transforms  

        train_dataset = YourDataset(root='./data', train=True, download=download, transform=transform)

3.create Dataloaders  

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  

4.training  

                for batch_idx,(data,labels) in enumerate(train_loader):

<!--                     data = data.to(device)
                    labels = labels.to(device), labels.to(device)

                    optimizer.zero_grad()
                    loss = criterion(output, labels)
                    loss.backward() 
                    optimizer.step()  -->

                output = model(data)



***  
# Modeling  
***  
# Training  
***  
# Evaluation  
***  
# Deployment  
***  
