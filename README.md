# mcux-sdk-middleware-eiq

This repository is for MCUXpresso SDK eIQ middleware delivery and it contains the components officially provided in NXP MCUXpresso SDK. 

This repository is a part of MCUXpresso SDK overall delivery which is composed of several project deliveries.
Please go to the [main repository](https://github.com/NXPmicro/mcux-sdk/) to get the MCUXpresso overall delivery to be able to build and run examples that are based on mcux-sdk-middleware-eiq components.

**The project is also the main repository to achieve the whole SDK eIQ delivery**, it contains the [west.yml](https://github.com/NXPmicro/mcux-sdk-middleware-eiq/blob/main/west.yml) which keeps description and revision for other components in the overall MCUXpresso eIQ delivery.
You need to have both Git and West installed, then execute below commands to achieve the whole SDK eIQ delivery at revision ```${revision}``` and place it in a folder named ```eiq```. 
```
west init -m https://github.com/NXPmicro/mcux-sdk-middleware-eiq --mr ${revision} eiq
cd eiq
west update
```
Replace ```${revision}``` with any SDK revision you wish to achieve. This can be ```MCUX_2.10.0``` or any commit SHA.
