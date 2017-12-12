/*
	u-pacs DICOM library
	Author : Y.Nagibovich
*/

#include "dcm_version.h"
#include "dcm_net.h"
#include "dcm_client.h"
#include "dcm_utils.h"
//#include "dcm_srv.h"
//#include "dcm_wsi.h"
//#include "dcm_file.h"

#include <string.h>
#include <stdio.h>
#include <direct.h>

//#include "DICOM_DataBase.h"

//#include "FolderUtils.h"
//#include "Iphlpapi.h"
//#include "nic.h"

#define CLEAR_BUFFER memset(buffer,0,MAX_BUFFER_SIZE/4)

BOOL DCM_CMD_FIND_SCU(char* buffer,  dicom_session_context_t* ctx,DCM_QWERY_t* qwery,DCM_QWERY_t* answer,int* answer_size,int mode)
{	
	int cmd_len=0,l,cnt;
	int buffer_size;
	int last_data,z,ans_cnt;
	DATA_SET_CMD_t* ptr;
	char* bptr;
	PDU_DATA_TF_EX_t* pdv;
	//DCM_QWERY_t answer[MAX_QRY_RESULT];
	char stg[128];
	CLEAR_BUFFER;
	// qry rq
	DCMAddCommand(buffer,&cmd_len,0x00000000,(UINT32)0);
	if(!mode) 
		DCMAddCommand(buffer,&cmd_len,0x00020000,QRY_FIND_STUDY);
	else 
		DCMAddCommand(buffer,&cmd_len,0x00020000,QRY_FIND_WL);
	DCMAddCommand(buffer,&cmd_len,0x01000000,(UINT16)0x0020); // find rq
	DCMAddCommand(buffer,&cmd_len,0x01100000, GetU16Odd(ctx)); // message id
	DCMAddCommand(buffer,&cmd_len,0x07000000,(UINT16)0x0); // proirity
	DCMAddCommand(buffer,&cmd_len,0x08000000,(UINT16)0x0102);
	DCMAddCommand(buffer,&cmd_len,0x00100002,DCM_FIND_TSYNTAX);
	DCMAddCommand(buffer,&cmd_len,0x00520008,qwery->s_mode.c_str());// Query Level
	for(cnt=0;cnt<qwery->Size();cnt++)
	{
		DCMAddCommand(buffer, &cmd_len, qwery->v_items[cnt].tag, qwery->v_items[cnt].data.c_str());
	}
	if(!DCMSendPDV(ctx,cmd_len,2)) // data
	{	
		if(ctx->event_exit) 
			SetEvent(ctx->event_exit);
		return FALSE; // data_set
	}
	if(!DCMSendPDU(ctx,buffer,cmd_len)) 
	{	
		if(ctx->event_exit) 
			SetEvent(ctx->event_exit);
		return FALSE;
	}
	// step 2
	cmd_len=0;
	// qery data
	CLEAR_BUFFER;
	DCMAddCommand(buffer,&cmd_len,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&cmd_len,0x00100002,DCM_FIND_TSYNTAX);
	DCMAddCommand(buffer,&cmd_len,0x00520008,qwery->s_mode.c_str());// Query Level
	for(cnt=0;cnt<qwery->Size();cnt++)
	{	
		DCMAddCommand(buffer, &cmd_len, qwery->v_items[cnt].tag, qwery->v_items[cnt].data.c_str());
	}
	if(!DCMSendPDV(ctx,cmd_len,2)) // data
	{	
		if(ctx->event_exit) 
			SetEvent(ctx->event_exit);
		return FALSE; // data_set
	}
	if(!DCMSendPDU(ctx,buffer,cmd_len)) 
	{	
		if(ctx->event_exit) 
			SetEvent(ctx->event_exit);
		return FALSE;
	}
	// receive ack
	ans_cnt=0;
	while(1)
	{	
		buffer_size=DCM_GET_DATA(buffer,ctx);

/*
	{	FILE* z;
		z=fopen("c:\\zzz","a+b");
		fwrite(buffer,buffer_size,1,z);
		fclose(z);

	}
*/
		if(buffer_size<0) 
			break;
		pdv=(PDU_DATA_TF_EX_t*)buffer;
		// reply analyzer
		for(l=0,cnt=0,last_data=0,bptr=buffer+12;l<buffer_size;)
		{	
			ptr=(DATA_SET_CMD_t*)bptr;
			if(ctx->assotiation_mode&ASS_EXPLICIT) 
				z=ptr->reserved;  
			else 
				z=ptr->val_size;
			if(pdv->tag&0x01) //cmd
			{	
				if(ptr->tag==0x08000000) 
				{	
					if(ptr->data==0x0101)
					{	
						last_data=1;
						break;
					}
				}
				else if(ptr->tag==0x01000000)
				{	
					if(ptr->data!=0x8020) // wrong reply
					{	
						last_data=1;
						break;
				    }
				}
			}
			else
			{	
				// TBD
				//if(ans_cnt<MAX_QRY_RESULT)
				{	
					memset(stg,0,128);
					memcpy(stg,&ptr->data,z&0x7F);
					//(answer+ans_cnt)->element[cnt].tag=ptr->tag;
					//(answer+ans_cnt)->element[cnt].data=stg;
					//(answer+ans_cnt)->size++;
					//if(cnt<(MAX_QRY-1)) 
					cnt++;
				}
			}
			bptr+=z+8;
			l+=z+8;
		}
		if(!(pdv->tag&0x01)) 
			ans_cnt++;
		if((pdv->tag==0x03) && last_data) 
			break;
	}
	if(answer_size) 
		*answer_size=ans_cnt;
	if(ctx->event_exit) 
		SetEvent(ctx->event_exit);
	return TRUE;
}

bool DCM_CMD_ECHO_SCU(char* buffer, dicom_session_context_t* ctx)
{	
	int cmd_len=0,l,cnt;
	int buffer_size;
	int last_data,z;
	bool retval=false;
	DATA_SET_CMD_t* ptr;
	char* bptr;
	PDU_DATA_TF_EX_t* pdv;
	CLEAR_BUFFER;
	// echo rq
	DCMAddCommand(buffer,&cmd_len,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&cmd_len,0x00020000,ECHO_SOP);
	DCMAddCommand(buffer,&cmd_len,0x01000000,(UINT16)0x0030); // echo rq
	DCMAddCommand(buffer,&cmd_len,0x01100000,(UINT16)GetU16Odd(ctx)); // message id
	DCMAddCommand(buffer,&cmd_len,0x07000000,(UINT16)0x0); // proirity
	DCMAddCommand(buffer,&cmd_len,0x08000000,(UINT16)0x0101); // no data
	//DCMAddCommand(buffer,&cmd_len,0x00100002,DCM_FIND_TSYNTAX);
	if(!DCMSendPDV(ctx,cmd_len,3)) 
	{	
		if(ctx->event_exit) 
			SetEvent(ctx->event_exit);
		return FALSE;
	}
	if(!DCMSendPDU(ctx,buffer,cmd_len)) 
	{	
		if(ctx->event_exit) 
			SetEvent(ctx->event_exit);
		return FALSE;
	}
	// receive ack
	while(1)
	{	
		buffer_size=DCM_GET_DATA(buffer,ctx);
		if(buffer_size<0) 
			break;
		pdv=(PDU_DATA_TF_EX_t*)buffer;
		// reply analyzer
		for(l=0,cnt=0,last_data=0,bptr=buffer+12;l<buffer_size;)
		{	
			ptr=(DATA_SET_CMD_t*)bptr;
			if(ctx->assotiation_mode&ASS_EXPLICIT) 
				z=ptr->reserved;  
			else 
				z=ptr->val_size;
			if(pdv->tag&0x01) //cmd
			{	
				if(ptr->tag==0x08000000) 
				{	
					if(ptr->data==0x0101)
					{	
						last_data=1;
						break;
					}
				}
				else if(ptr->tag==0x01000000)
				{	
					if(ptr->data!=0x8030) 
						retval=FALSE; 
					else 
						retval=TRUE; // reply is ok
					last_data=1;
					break;
				}
			}
			bptr+=z+8;
			l+=z+8;
		}
		if((pdv->tag==0x03) && last_data) 
			break;
	}
	if(ctx->event_exit) 
		SetEvent(ctx->event_exit);
	return retval;
}

bool DCM_CMD_STORE_SCU(char* buffer, dicom_session_context_t* ctx, image_store_info_t* storage, int mode)
{	
	int l=0, z, i, ret,total,offset;
	int buffer_size;
	PDU_DATA_TF_EX_t* dptr;
	char* bptr=buffer;
	char* sop;
	bool retval=TRUE;

#ifdef DICOM_SERVER
	if(ctx->db_ctx && (!ctx->db_ctx->get_storage_info_cb(ctx, storage)))
	{
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
#endif //DICOM_SERVER
	if(storage->sop_UID.size()) sop = (char*)storage->sop_UID.c_str();
	else sop=STORAGE_CT_IMAGE;
	// write store-rq
	CLEAR_BUFFER;
	DCMAddCommand(buffer,&l,0x00000000, (UINT32)0);
	DCMAddCommand(buffer,&l,0x00020000, sop);
	DCMAddCommand(buffer,&l,0x01000000, (UINT16)0x0001);
	DCMAddCommand(buffer,&l,0x01100000, (UINT16)GetU16Odd(ctx)); // message id
	DCMAddCommand(buffer,&l,0x07000000, (UINT16)0x0002); // priority
	if(!mode)
	{
		DCMAddCommand(buffer,&l,0x08000000, (UINT16)0x0001); // data set
		DCMAddCommand(buffer,&l,0x10000000, storage->image_UID.c_str()); // 
	}
	else
	{
		DCMAddCommand(buffer,&l,0x08000000, (UINT16)0x0102); // data set
		DCMAddCommand(buffer,&l,0x10000000, storage->image_UID.c_str()); // 
		DCMAddCommand(buffer,&l,0x10300000, ctx->assotiation_calling_AET);
		DCMAddCommand(buffer,&l,0x10310000, (UINT16)ctx->parent_message_id); // message id
	}
	if(!DCMSendPDV(ctx,l,3)) // last 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	if(!DCMSendPDU(ctx, buffer, l)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	// send file
	i=MAX_BUFFER_SEND_SIZE-sizeof(PDU_DATA_TF_EX_t);
	while(!feof(storage->datafile))
	{	
		l=fread(buffer, 1, i, storage->datafile);
		if(l<i) 
		{	
			z=0x02;
		}
		else z=0;
		if(!DCMSendPDV(ctx, l, z)) 
		{	
			retval=FALSE;
			break;
		}
		if(!DCMSendPDU(ctx,buffer,l,1)) 
		{	
			retval=FALSE;
			break;
		}
	}
	// get rq
	memset(buffer,0,512);
	l=0;
	// receive ack
	while(!l)
	{	
		buffer_size=FIRST_PACKET;
		ret=DCMRecvData(ctx,buffer,buffer_size);
		if(ret==SOCKET_ERROR) break;
		dptr=(PDU_DATA_TF_EX_t*)buffer;
		if(dptr->type==0x05) // assoc release rq
		{	
			break;
		}
		if(dptr->type==0x07) // assoc abort
		{	
			retval=FALSE;
			break;
		}
		sb(&dptr->length); // correct endianing
		buffer_size=dptr->length-ret+4+2; 
		if(buffer_size>MAX_BUFFER_SIZE) 
		{	
			retval=FALSE;
			break;
		}
		if(buffer_size<=0) 
		{	
			retval=FALSE;
			break;
		}
		offset=ret;total=0;
		while(total<buffer_size)
		{	
			ret=DCMRecvData(ctx,buffer+total+offset,buffer_size-total);
			total+=ret;
			if(ret==SOCKET_ERROR) 
			{	
				retval=FALSE;
				break;
			}
		}
		//ret=recv(ctx->socket,buffer+ret,buffer_size,0);
		if(ret==SOCKET_ERROR) 
		{	
			retval=FALSE;
			break;
		}
		l=1;
	}
	if(retval && ctx->logger)
	{
		ctx->logger( DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM сессия #%d. STORE_SCU - OK. [UID: %s]", ctx->session_id, storage->image_UID.c_str());
	}
	if(storage->datafile)
	{
		fclose(storage->datafile); // close file handler
		storage->datafile=NULL;
	}
	if(ctx->event_exit) 
		SetEvent(ctx->event_exit);
	return retval;
}
/*

BOOL DCM_CMD_MOVE_SCU(char* buffer, dicom_session_context_t* ctx,char* destination_AET,DCM_QWERY_t* qwery)
{	
	int cmd_len=0,cnt;
	int move_remaining;
	int move_completed;
	int move_failed;
	int move_warning;
	int l,z,last_data=0;
	int buffer_size,total=0;
	DATA_SET_CMD_t* ptr;
	PDU_DATA_TF_EX_t* pdv;
	char* bptr;
	
	CLEAR_BUFFER;
	// qry rq
	DCMAddCommand(buffer,&cmd_len,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&cmd_len,0x00020000,QRY_MOVE_STUDY);
	DCMAddCommand(buffer,&cmd_len,0x01000000,(UINT16)0x0021); // move rq
	DCMAddCommand(buffer,&cmd_len,0x01100000,GetU16Odd(ctx)); // message id
	DCMAddCommand(buffer,&cmd_len,0x06000000,destination_AET);
	DCMAddCommand(buffer,&cmd_len,0x07000000,(UINT16)0x0);
	DCMAddCommand(buffer,&cmd_len,0x08000000,(UINT16)0x0102);
	DCMAddCommand(buffer,&cmd_len,0x00100002,DCM_FIND_TSYNTAX);
	DCMAddCommand(buffer,&cmd_len,0x00520008,qwery->mode.c_str());// Query Level
	for(cnt=0;cnt<qwery->size;cnt++)
	{	
		DCMAddCommand(buffer,&cmd_len,qwery->element[cnt].tag,qwery->element[cnt].data.c_str());
	}
	if(!DCMSendPDV(ctx,cmd_len,2)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	if(!DCMSendPDU(ctx,buffer,cmd_len)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	// step2
	cmd_len=0;
	// qwery data
	CLEAR_BUFFER;
	DCMAddCommand(buffer,&cmd_len,0x00000000,(UINT32)0);
	DCMAddCommand(buffer,&cmd_len,0x00100002,DCM_FIND_TSYNTAX);
	DCMAddCommand(buffer,&cmd_len,0x00520008,qwery->mode.c_str());// Query Level
	for(cnt=0;cnt<qwery->size;cnt++)
	{	
		DCMAddCommand(buffer,&cmd_len,qwery->element[cnt].tag,qwery->element[cnt].data.c_str());
	}
	if(!DCMSendPDV(ctx,cmd_len,2)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE; // data_set
	}
	if(!DCMSendPDU(ctx,buffer,cmd_len)) 
	{	
		if(ctx->event_exit) SetEvent(ctx->event_exit);
		return FALSE;
	}
	// receive ack
	move_remaining=0;
	move_completed=0;
	move_failed=0;
	move_warning=0;
	while(1)
	{	
		buffer_size=DCM_GET_DATA(buffer,ctx);
		if(buffer_size<0) break;
		pdv=(PDU_DATA_TF_EX_t*)buffer;
		// reply analyzer
		for(l=0,last_data=0,bptr=buffer+12;l<buffer_size;)
		{	
			ptr=(DATA_SET_CMD_t*)bptr;
			if(ctx->assotiation_mode&ASS_EXPLICIT) z=ptr->reserved;  
			else z=ptr->val_size;
			if(pdv->tag&0x01) //cmd
			{	
				if(ptr->tag==0x01000000)
				{	
					if(ptr->data!=0x8021) // wrong reply
					{	
						last_data=1;
						break;
					}
					else
					{	
						last_data=0;
					}
				}
			}
			//else
			{	
				switch(ptr->tag)
				{	
					case 0x10200000 : move_remaining=ptr->data;break; // remaining
					case 0x10210000 : move_completed=ptr->data;break; // completed cstore
					case 0x10220000 : move_failed=ptr->data;break; // failed cstore
					case 0x10230000 : move_warning=ptr->data;break;// warning cstore
					case 0x09000000 : if(ptr->data==0x00) last_data=1;break;
				}
			}
			bptr+=z+8;
			l+=z+8;
		}
#ifndef DICOM_SERVER
		if(ctx->event_step) SetEvent(ctx->event_step);
#endif
		if((pdv->tag==0x03) && last_data) break;
	}
	// use external 
	if(ctx->event_exit) SetEvent(ctx->event_exit);
	return TRUE;
}
*/
/*
DWORD WINAPI DCMSessionProc(PVOID pParameter)
{	
	dicom_session_context_t* ctx;
	ASSOCIATE_RQ_t* arq;
	PDU_DATA_TF_EX_t*  cmd;
	int total,offset;
	char session_buffer[MAX_BUFFER_SIZE];
	int buffer_size;
	int ret,err;
	int ass_done=0;
	byte val;
	
	ctx=( dicom_session_context_t*)pParameter;
	while(1)
	{	
		// start association
		buffer_size=FIRST_PACKET;
		//ret=recv(ctx->socket,session_buffer,buffer_size,0);
		ret=DCMRecvData(ctx,session_buffer,buffer_size);
		if(ret==SOCKET_ERROR) 
		{	
			err=WSAGetLastError();
			break;
		}
		arq=(ASSOCIATE_RQ_t*)session_buffer;
		sb(&arq->length); // correct endianing
		buffer_size=arq->length-ret+4+2; //??
		if(buffer_size>MAX_BUFFER_SIZE) 
		{	
			break;
		}
		if(buffer_size<=0) 
		{	
			break;
		}
		offset=ret;total=0;
		while(total<buffer_size)
		{	
			//ret=recv(ctx->socket,session_buffer+total+offset,buffer_size-total,0);
			ret=DCMRecvData(ctx,session_buffer+total+offset,buffer_size-total);
			total+=ret;
			if(ret==SOCKET_ERROR) 
			{	
				break;
			}
		}
		if(ret==SOCKET_ERROR) 
		{	
			break;
		}
		// analyze association
		if(!StartAssociation(session_buffer,arq->length+4,ctx)) break;
		ass_done=1;
		break;
	}
	while(ass_done)
	{	
		buffer_size=FIRST_PACKET;
		//ret=recv(ctx->socket,session_buffer,buffer_size,0);
		ret=DCMRecvData(ctx,session_buffer,buffer_size);
		if(ret==SOCKET_ERROR) 
		{	
			break;
		}
		if(session_buffer[0]==0x07) // assoc abort
		{	
			if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM session #%d aborted by peer", ctx->session_id);
			break;
		}
		if(session_buffer[0]==0x05) // assoc release
		{	
			if(!DCMEndAssociationRply(ctx,session_buffer)) 
			{	
				break;
			}
			// close session
			if(ctx->status==SESSION_STOPPED) 
			{	
				break;
			}
		}
		cmd=(PDU_DATA_TF_EX_t*)session_buffer;
		sb(&cmd->length); // correct endianing
		buffer_size=cmd->length-ret+4+2; 
		if(buffer_size>MAX_BUFFER_SIZE) 
		{	
			break;
		}
		if(buffer_size<=0) 
		{	
			break;
		}
		offset=ret;total=0;
		while(total<buffer_size)
		{	
			ret=DCMRecvData(ctx,session_buffer+total+offset,buffer_size-total);
			total+=ret;
			if(ret==SOCKET_ERROR) 
			{	
				break;
			}
		}
		if(ret==SOCKET_ERROR) 
		{	break;
		}
		if(cmd->type!=0x04) 
		{	
			if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_ERROR, "DICOM session #%d aborted. Wrong data.", ctx->session_id);
			break;
		}
		// handle ass endianing & etc
		ctx->association_PID=*(session_buffer+10);
		ctx->assotiation_mode=0;
		val=ctx->association[(ctx->association_PID>>1)&0x7F].type;
		switch(val)
		{	
			case 0 : ctx->assotiation_mode=0;break;
			case 1 : 
			case 2 : ctx->assotiation_mode=ASS_EXPLICIT;break;
			case 3 : ctx->assotiation_mode=ASS_EXPLICIT|ASS_BIGENDIAN;break;
			default : ctx->assotiation_mode=ASS_EXPLICIT;break;
		}
		if(!AnalyzeCommand(session_buffer, ctx)) break;
	}
	DCMCloseConnection(ctx, DCM_SERVER);
	// log
	if(ctx->logger) ctx->logger(DCM_LOG_NETWORK_EVENT, DCM_LOG_OK, "DICOM session #%d closed", ctx->session_id);
	// free session
	DCMRemoveSession(ctx);
	return 0;
}
*/
