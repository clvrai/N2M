#include <stdio.h>
#include <rclcpp/rclcpp.hpp> 
#include <curl/curl.h>
#include "../thirdparty/rapidjson/document.h"
#include "../thirdparty/rapidjson/prettywriter.h"
#include "../include/remote.h"

using namespace rapidjson;
rclcpp::Node::SharedPtr node_handle = nullptr;
static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

static size_t dummy_callback(void *buffer, size_t size, size_t nmemb, void *userp)
{
   return size * nmemb;
}

int sensor_config(std::string sensor_ipaddr, std::string parameter, std::string value)
{
	RCLCPP_INFO(node_handle->get_logger(),"URL_RESTFUL_API");
	
	long http_code;
	CURL *curl = curl_easy_init();
	std::string URL_RESTFUL_API = "http://" + sensor_ipaddr + parameter;
	RCLCPP_INFO(node_handle->get_logger(),"URL_RESTFUL_API%s",URL_RESTFUL_API.c_str());
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3);
	if(curl) {
		curl_easy_setopt(curl, CURLOPT_URL, URL_RESTFUL_API.c_str());
    	curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, value.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, dummy_callback);
		if (curl_easy_perform(curl) == CURLE_OK){
			curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
			if(http_code == 200)
			{
				RCLCPP_INFO(node_handle->get_logger(),"Set %s, Value: %s ... done", URL_RESTFUL_API.c_str(), value.c_str());
			}
			else
			{
				RCLCPP_INFO(node_handle->get_logger(),"Set %s, Value: %s ... failed!", URL_RESTFUL_API.c_str(), value.c_str());
			}
		}
		else
		{
			RCLCPP_INFO(node_handle->get_logger(),"http put error! please check lidar connection!");
		}
	}
	curl_easy_cleanup(curl);
    curl_global_cleanup();

	return 0;
}

int get_telemetry_data(std::string sensor_ipaddr)
{
	CURL *curl;
	std::string readBuffer;
	std::string URL_API_FIRMWARE = "http://" + sensor_ipaddr + "/api/v1/system/firmware";
	std::string URL_API_MONITOR = "http://" + sensor_ipaddr + "/api/v1/system/monitor";
	std::string URL_API_OVERVIEW = "http://" + sensor_ipaddr + "/api/v1/sensor/overview";

	curl = curl_easy_init();
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3);
	if(curl) {
		readBuffer = "";
		curl_easy_setopt(curl, CURLOPT_URL, URL_API_FIRMWARE.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		curl_easy_perform(curl);
		curl_easy_cleanup(curl);
		const char* json = const_cast<char*>(readBuffer.c_str());
		Document jsondoc;
		jsondoc.Parse(json);
		assert(jsondoc.IsObject());
		RCLCPP_INFO(node_handle->get_logger(),"-------------------------------------------------");
		RCLCPP_INFO(node_handle->get_logger(),"model:		%s", jsondoc["model"].GetString());
		RCLCPP_INFO(node_handle->get_logger(),"sn:		%s", jsondoc["sn"].GetString());
		RCLCPP_INFO(node_handle->get_logger(),"ver hw:		%s", jsondoc["hw"].GetString());
		RCLCPP_INFO(node_handle->get_logger(),"ver fpga:	%s", jsondoc["fpga"].GetString());
		RCLCPP_INFO(node_handle->get_logger(),"ver core:	%s", jsondoc["core"].GetString());
		RCLCPP_INFO(node_handle->get_logger(),"ver aux:	%s", jsondoc["aux"].GetString());
	}

	curl = curl_easy_init();
	if(curl) {
		readBuffer = "";
		curl_easy_setopt(curl, CURLOPT_URL, URL_API_MONITOR.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		curl_easy_perform(curl);
		curl_easy_cleanup(curl);

		const char* json = const_cast<char*>(readBuffer.c_str());
		Document jsondoc;
		jsondoc.Parse(json);
		assert(jsondoc.IsObject());
		RCLCPP_INFO(node_handle->get_logger(),"load average:	%.2f", jsondoc["load_average"].GetDouble());
		RCLCPP_INFO(node_handle->get_logger(),"men useage:	%.2f", jsondoc["mem_useage"].GetDouble());
		RCLCPP_INFO(node_handle->get_logger(),"uptime:		%.2f sec", jsondoc["uptime"].GetDouble());
	}

	curl = curl_easy_init();
	if(curl) {
		readBuffer = "";
		curl_easy_setopt(curl, CURLOPT_URL, URL_API_OVERVIEW.c_str());
		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
		curl_easy_perform(curl);
		curl_easy_cleanup(curl);

		const char* json = const_cast<char*>(readBuffer.c_str());
		Document jsondoc;
		jsondoc.Parse(json);
		assert(jsondoc.IsObject());
		RCLCPP_INFO(node_handle->get_logger(),"scanfreq:	%d hz", jsondoc["scanfreq"].GetInt());
		RCLCPP_INFO(node_handle->get_logger(),"motor rpm:	%d (%.2fhz)", jsondoc["motor_rpm"].GetInt(), (jsondoc["motor_rpm"].GetInt() / 60.f));
		RCLCPP_INFO(node_handle->get_logger(),"laser enable:	%d", jsondoc["laser_enable"].GetBool());
		RCLCPP_INFO(node_handle->get_logger(),"scan start:	%d deg", jsondoc["scan_range"]["start"].GetInt());
		RCLCPP_INFO(node_handle->get_logger(),"scan stop:	%d deg", jsondoc["scan_range"]["stop"].GetInt());
		RCLCPP_INFO(node_handle->get_logger(),"flt level:	%d", jsondoc["filter"]["level"].GetInt());
		RCLCPP_INFO(node_handle->get_logger(),"flt min_angle:	%d", jsondoc["filter"]["min_angle"].GetInt());
		RCLCPP_INFO(node_handle->get_logger(),"flt max_angle:	%d", jsondoc["filter"]["max_angle"].GetInt());
		RCLCPP_INFO(node_handle->get_logger(),"flt window:	%d", jsondoc["filter"]["window"].GetInt());
		RCLCPP_INFO(node_handle->get_logger(),"flt neighbors:	%d", jsondoc["filter"]["neighbors"].GetInt());
		RCLCPP_INFO(node_handle->get_logger(),"host ip:	%s", jsondoc["host"]["ip"].GetString());
		RCLCPP_INFO(node_handle->get_logger(),"host port:	%d", jsondoc["host"]["port"].GetInt());
		RCLCPP_INFO(node_handle->get_logger(),"-------------------------------------------------");
	}

	return 0;
}