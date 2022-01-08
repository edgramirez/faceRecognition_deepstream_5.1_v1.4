import lib.common as com


def check_config_keys_exist(service_name, service_dictionary):
    joint_elements = []
    for group_of_parameters in com.SERVICE_DEFINITION[com.SERVICES[service_name]][service_name]:
        for defined_service_parameter in com.SERVICE_DEFINITION[com.SERVICES[service_name]][service_name][group_of_parameters]:
            joint_elements.append(defined_service_parameter)
    for service_parameter in service_dictionary:
        if service_parameter not in joint_elements:
            com.log_error("Configuration error - Pameter: {}, does not exist in the service definition:".format(service_parameter))
    return True


def validate_sources(active_service_configs):
    '''
    Validate the configuration source recovered from server contains correct values
    '''
    for camera_service_id in active_service_configs:
        for service in active_service_configs[camera_service_id]:
            pattern_not_found = True
            for pattern in com.SOURCE_PATTERNS:
                if active_service_configs[camera_service_id][service]['source'][0:len(pattern)] == pattern:
                    if active_service_configs[camera_service_id][service]['source'][:7] == 'file://' and com.file_exists(active_service_configs[camera_service_id][service]['source'][7:]) is False:
                        com.log_error("Configuration error - Source file: {}, does not exist".format(active_service_configs[camera_service_id][service]['source'][7:]))
                    pattern_not_found = False
                    break
            if pattern_not_found:
                com.log_error("Configuration error - Source value must start with any of this patterns: {}, Current value: {}".format(com.SOURCE_PATTERNS, active_service_configs[camera_service_id][service]['source']))

    com.log_debug('All source values are correct')
    return True


def check_obligatory_keys(service_dictionaries, service_definition):
    '''
    Validate the configuration recovered from server provided the defined minimum parameters and their values are valid
    '''
    for defined_item in service_definition['obligaroty'].keys():
        for service_name in service_dictionaries:
            if defined_item not in service_dictionaries[service_name]:
                com.log_error("Configuration error - Missing Obligatory parameter: {}".format(defined_item))
            if str(type(service_dictionaries[service_name][defined_item])).split("'")[1] != service_definition['obligaroty'][defined_item]:
                com.log_error("Configuration error - Parameter '{}' value must be type : {}, Current value: {}".format(defined_item, service_definition['obligaroty'][defined_item], str(type(service_dictionaries[defined_item])).split("'")[1]))
    com.log_debug("All obligatory parameters are OK")
    return True


def check_optional_keys(service_dictionaries, service_definition):
    '''
    Validate the optional configuration recovered from server and its values
    '''
    for defined_item in service_definition['optional'].keys():
        for service_name in service_dictionaries:
            if defined_item in service_dictionaries[service_name] and str(type(service_dictionaries[service_name][defined_item])).split("'")[1] != service_definition['optional'][defined_item]:
                    com.log_error("Configuration error - Parameter '{}' value must be type : {}, Current value: {}".format(defined_item, service_definition['optional'][defined_item], str(type(service_dictionaries[service][defined_item])).split("'")[1]))

    com.log_debug("All optional parameters are OK")
    return True


def check_service_against_definition(data):
    if not isinstance(data, dict):
        com.log_error("Configuration error - data must be a list of dictionaries - type: {} / content: {}".format(type(data), data))

    for srv_camera_id in data.keys():
        for service in data[srv_camera_id].keys():
            com.log_debug("Validating config of service: '--{} / {}--' against its coded definition: \n\n{}\n\n".format(service, srv_camera_id, data[srv_camera_id][service]))
            for parameters in data[srv_camera_id]:
                check_config_keys_exist(parameters, data[srv_camera_id][parameters])
            check_obligatory_keys(data[srv_camera_id], com.SERVICE_DEFINITION[com.SERVICES[service]][service])
            check_optional_keys(data[srv_camera_id], com.SERVICE_DEFINITION[com.SERVICES[service]][service])
    return True


def validate_service_exists(data):
    for camera_service_id in data.keys():
        for service_name in data[camera_service_id].keys():
            if service_name not in com.SERVICES:
                com.log_error("Configuration error - Requested service: {} - Does not exist in the service list: {}".format(service_name, com.SERVICES))
    return True


def get_config_filtered_by_active_service(config_data):
    if not isinstance(config_data, dict):
        com.log_error("Configuration error - Config data must be a dictionary - type: {} / content: {}".format(type(config_data), config_data))
    active_services = {}

    for local_server_mac in config_data.keys():
        # This variable will be incremented if the service name key already exists
        for camera_mac in config_data[local_server_mac]:
            for service in config_data[local_server_mac][camera_mac]:
                if 'enabled' in config_data[local_server_mac][camera_mac][service] and config_data[local_server_mac][camera_mac][service]['enabled'] is True:
                    # Create new key only for the active service
                    new_key_name = 'srv_' + local_server_mac + "_camera_" + camera_mac + '_' + service
                    active_services[new_key_name] = {service: config_data[local_server_mac][camera_mac][service]}

    if len(active_services) < 1:
        com.log_error("\nConfiguration does not contain any active service for this server: \n\n{}".format(config_data))

    return active_services


def mac_address_in_config(mac_config):
    for machine_id in com.get_machine_macaddresses():
        if mac_config == machine_id:
            return True
    return False


def get_config_filtered_by_local_mac(config_data):
    '''
    By now we only support one nano server and one interface 
    but it can be a big server with multiple interfaces so I 
    leave the logic with to handle this option
    '''
    services_data = {}
    for key in config_data.keys():
        if mac_address_in_config(key):
            services_data[key] = config_data[key]
    if services_data:
        return services_data

    com.log_error("The provided configuration does not match any of server interfaces mac address")


def parse_parameters_and_values_from_config(config_data):
    # filter config and get only data for this server using the mac to match
    scfg = get_config_filtered_by_local_mac(config_data)

    # filter config and get only data of active services
    scfg = get_config_filtered_by_active_service(scfg)

    # validate requested services exists in code
    validate_service_exists(scfg)

    # Check all obligatory and optional parameters and values types provided by the dashboard config
    check_service_against_definition(scfg)

    # Check all source values to ensure they are correct and in the case of files they actually exists
    validate_sources(scfg)

    return scfg

