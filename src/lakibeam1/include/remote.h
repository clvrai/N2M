#ifndef __REMOTE_H__
#define __REMOTE_H__

#include <stdio.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
int sensor_config(std::string sensor_ipaddr, std::string parameter, std::string value);
int get_telemetry_data(std::string sensor_ipaddr);


#endif