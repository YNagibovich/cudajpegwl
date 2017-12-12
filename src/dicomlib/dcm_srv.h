/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#pragma once

// DICOM library specific headers
#include "windows.h"
#include "winsock2.h"
#include "dcm_version.h"
#include "dcm_data.h"
#include "dcm_constants.h"
#include "dcm_qry.h"

#include <vector>

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

// storage info
typedef struct  
{
	unsigned int image_id;
	FILE* datafile;
	std::string sop_UID;
	std::string image_UID;
} image_store_info_t;

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

typedef struct  
{
	SOCKET socket; /// main listening socket
	std::vector<dicom_session_context_t*> v_sessions;
	//session_context_t* sessions[MAX_DICOM_SESSIONS];
	// own settings
	char aet[32];
	int port;
	int nic_idx;
	int state;
	HANDLE main_thread;
	int timeout;
	char* own_guid;
	//CCriticalSection* srv_lock;
	// callbacks
	void* db_conn;
	db_handlers_t* db_ctx;
	dcm_logger_cb logger;
	dcm_dumper_cb dumper;
} dicom_server_context_t;

// export functions
int GetSOPId(char* sop);

// helpers
unsigned long get_inet_addr(char* host_name);

// internals
bool CreateAssociacion(char* buffer, dicom_session_context_t* ctx, char* p_context, int t_syntax);
bool DCMEndAssociationGetRply( dicom_session_context_t* ctx, char* buf);
bool DCMEndAssociationRply( dicom_session_context_t* ctx, char* buf);
bool DCMEndAssociationRq( dicom_session_context_t* ctx, char* buf);
bool DCM_CMD_FIND_SCU(char* buffer, dicom_session_context_t* ctx, 
					  DCM_QWERY_t* qwery, DCM_QWERY_t* answer, int* answer_size, int mode=0);
bool DCM_CMD_ECHO_SCU(char* buffer, dicom_session_context_t* ctx);
bool DCM_CMD_MOVE_SCU(char* buffer, dicom_session_context_t* ctx, char* destination_AET, DCM_QWERY_t* qwery);
bool DCM_CMD_STORE_SCU(char* buffer, dicom_session_context_t* ctx, image_store_info_t* storage, int mode);

// server control
// create server context
dicom_server_context_t* DICOMServerCreate(DICOM_PROVIDER_t* settings, 
										  dcm_logger_cb logger_cb,
										  dcm_dumper_cb dumper_cb,
										  db_handlers_t* db_ctx,
										  int timeout, 
										  int nic_idx=0);
// start main server thread
bool DICOMServerStart( dicom_server_context_t* pctx);

// stop main server thread
bool DICOMServerStop( dicom_server_context_t* pctx);

// destroy server context
bool DICOMServerClose( dicom_server_context_t* pctx);

// sent progress notifications to parent window
#define WM_USER_DCIOM_BASE					(WM_APP+200)
#define WM_USER_DCIOM_ECHO_START			WM_USER_DCIOM_BASE+1
#define WM_USER_DCIOM_ECHO_STEP				WM_USER_DCIOM_BASE+2
#define WM_USER_DCIOM_ECHO_STOP				WM_USER_DCIOM_BASE+3

#define WM_USER_DCIOM_SEND_START			WM_USER_DCIOM_BASE+4
#define WM_USER_DCIOM_SEND_STEP				WM_USER_DCIOM_BASE+5
#define WM_USER_DCIOM_SEND_STOP				WM_USER_DCIOM_BASE+6

// High level, SCU
bool DICOMEchoe(void* parent, std::string sAET, DICOM_PROVIDER_t* info, dcm_logger_cb cb_logger);
bool DICOMSend(void* parent, std::string sAET, DICOM_PROVIDER_t* info, const char* file_name, dcm_logger_cb cb_logger);