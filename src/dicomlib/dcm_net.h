/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#pragma once

// DICOM library specific headers
//#include "windows.h"
#include "winsock2.h"

#include "dcm_version.h"
#include "dcm_data.h"
#include "dcm_constants.h"
#include "dcm_qry.h"

// association modes
#define ASS_EXPLICIT  0x01
#define ASS_BIGENDIAN 0x02
#define ASS_SUPPORTED 0x00
#define ASS_NOT_SUPPORTED 0x04

// aux
#define SESSION_STARTED 1
#define SESSION_STOPPED 0
#define RECONNECT_COUNT 3
#define RECONNECT_TIMEOUT 150

// limits
#define MAX_BUFFER_SIZE			(1024*256)
#define MAX_BUFFER_SEND_SIZE	(1024*16)
#define FIRST_PACKET			10 // PDV size -2
#define MAX_ASSOCIATION			128
#define MAX_DICOM_SESSIONS		1024

// callbacks

/*
*	logger
*/
//typedef void (*dcm_logger_cb)(int log_mode, int log_level, int type, int result, unsigned int subsys_id, char* fmt, ...);
typedef void (*dcm_logger_cb)(int type, int result, char* fmt, ...);

/*
*	logger
* @param type -  : 0 - incoming, 1 outgoing
* @param len - data length
* @param data - data buffer
*/
typedef void (*dcm_dumper_cb)(int session_id, int type, int len, char* data);

// DB handlers

/*
*	find DICOM partner
* @param aet - calling AET
* @return address or NULL if not found
*/
typedef char* (*db_partner_find_addr_cb)(void* user_ctx, char* aet);
typedef int (*db_partner_find_port_cb)(void* user_ctx, char* aet);

/*
*	find partner's arc mode
* @param aet - calling AET
* @return arc mode
*/
typedef int (*db_partner_arc_mode_cb)(void* user_ctx, char* aet);

typedef char* (*db_create_image_path_cb)(void* user_ctx, char* calling_aet, PATIENT_DATA_t* pd, EXAM_DATA_t* ed, SERIES_DATA_t* sd, IMAGE_DATA_t* imd);

typedef int (*db_find_image_cb)(void* user_ctx, IMAGE_DATA_t* image);

typedef int (*db_add_data_cb)(void* user_ctx, PATIENT_DATA_t* patient, EXAM_DATA_t* exam, SERIES_DATA_t* series, IMAGE_DATA_t* image);

typedef int (*db_qwery_DWL_cb)(void* user_ctx, DCM_QWERY_t* qry, DCM_QWERY_t* answer);

typedef int (*db_qwery_cb)(void* user_ctx, DCM_QWERY_t* qry, DCM_QWERY_t* answer);

//typedef char* (*db_get_image_ID_cb) (void* user_ctx, int image_id);
//typedef char* (*db_get_image_path_cb)(void* user_ctx, int image_id);
//typedef char* (*db_get_image_SopUID_cb)(void* user_ctx, int image_id);

typedef int (*db_get_storage_info_cb)(void* user_ctx, void* ctx);

typedef int (*db_get_files_cb)(void* user_ctx, DCM_QWERY_t* qry, std::vector<uint64_t> output, int* t_syntax);

typedef int (*db_set_connection_cb)(void* user_ctx);

typedef int (*db_close_connection_cb)(void* user_ctx);

typedef char* (*db_conn_get_server_name)(void* user_ctx);
typedef char* (*db_conn_get_base_name)(void* user_ctx);
typedef char* (*db_conn_get_user_login)(void* user_ctx);
typedef char* (*db_conn_get_user_password)(void* user_ctx);

typedef struct  
{	
	// db handlers
	db_set_connection_cb set_connection_cb;
	db_close_connection_cb close_connection_cb;
	db_partner_find_addr_cb partner_find_addr_cb;
	db_partner_find_port_cb partner_find_port_cb;
	db_partner_arc_mode_cb partner_arc_mode_cb;
	db_create_image_path_cb create_image_path_cb;
	db_find_image_cb find_image_cb;
	db_add_data_cb add_data_cb;
	db_qwery_DWL_cb qwery_DWL_cb;
	db_qwery_cb qwery_cb;
	db_get_storage_info_cb get_storage_info_cb;
	//db_get_image_ID_cb get_image_ID_cb;
	//db_get_image_path_cb get_image_path_cb;
	//db_get_image_SopUID_cb get_image_SopUID_cb;
	db_conn_get_server_name conn_get_server_name;
	db_conn_get_base_name conn_get_base_name;
	db_conn_get_user_login conn_get_user_login;
	db_conn_get_user_password conn_get_user_password;
	db_get_files_cb get_files_cb;
} db_handlers_t;

// internal structures
typedef struct  
{	
	SOCKET socket; /// session socket
	int status;
	HANDLE session;
	int8_t association_PID;
	int  assotiation_mode;
	char assotiation_calling_AET[17];
	char* remote_aet;
	char* remote_addr;
	char* own_guid;
	unsigned short remote_port;
	unsigned int session_id;
	int parent_message_id;
	ASSOCIATION_t association[MAX_ASSOCIATION];
	int arc_mode;
	int end_ass_rq;
	// helpers
	HANDLE event_exit;
	HANDLE event_step;
	void* parent;
	// UID
	unsigned char U_8;
	unsigned short U_16;
	unsigned int U_32;
	int timeout;
	int err_code;
	void* db_conn;
	// callbacks
	db_handlers_t* db_ctx;
	dcm_logger_cb logger;
	dcm_dumper_cb dumper;
} dicom_session_context_t;

// export functions
int GetSOPId(uint8_t* sop);
void ResetUID(dicom_session_context_t* ctx);

// helpers
unsigned long get_inet_addr(char* host_name);
int DCM_GET_DATA(char* buffer, dicom_session_context_t* ctx);
int8_t GetU8(dicom_session_context_t* ctx);
UINT16 GetU16(dicom_session_context_t* ctx);
UINT16 GetU16Odd(dicom_session_context_t* ctx);
UINT32 GetU32(dicom_session_context_t* ctx);

void ClearDCMQwery(DCM_QWERY_t* qry);
bool DCM_ADD_QRY_TAG(DCM_QWERY_t* qry, uint32_t tag);
bool DCM_INIT_QRY_TAGS(DCM_QWERY_t* qry, uint8_t* sop_uid);
bool DCM_SET_QRY_TAG(DCM_QWERY_t* qry, uint32_t tag, uint8_t* val);

// internals
bool CreateAssociacion(char* buffer, dicom_session_context_t* ctx, char* p_context, int t_syntax);
bool DCMEndAssociationGetRply(dicom_session_context_t* ctx, char* buf);
bool DCMEndAssociationRply(dicom_session_context_t* ctx, char* buf);
bool DCMEndAssociationRq(dicom_session_context_t* ctx, char* buf);

void DCMAddCommand(char* buffer, int* cmd_length, uint32_t tag, UINT16 data);
void DCMAddCommand(char* buffer, int* cmd_length, uint32_t tag, uint32_t data);
void DCMAddCommand(char* buffer, int *cmd_length, uint32_t tag, const char* data);

bool DCMSendPDU(dicom_session_context_t* ctx, char* buf, uint32_t len, int mode = 0);
bool DCMSendPDV(dicom_session_context_t* ctx, uint32_t len, int8_t tag);

int DCMRecvData(dicom_session_context_t* ctx, char* buf, UINT32 len);