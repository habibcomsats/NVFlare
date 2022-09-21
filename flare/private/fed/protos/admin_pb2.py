# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: flare/private/fed/protos/admin.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='flare/private/fed/protos/admin.proto',
  package='admin',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n&flare/private/fed/protos/admin.proto\x12\x05\x61\x64min\"+\n\x08Messages\x12\x1f\n\x07message\x18\x01 \x03(\x0b\x32\x0e.admin.Message\"\xa3\x01\n\x07Message\x12\n\n\x02id\x18\x01 \x01(\t\x12\r\n\x05topic\x18\x02 \x01(\t\x12\x11\n\tbody_type\x18\x03 \x01(\t\x12\x0c\n\x04\x62ody\x18\x04 \x01(\x0c\x12,\n\x07headers\x18\x05 \x03(\x0b\x32\x1b.admin.Message.HeadersEntry\x1a.\n\x0cHeadersEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"\x07\n\x05\x45mpty\"\x1d\n\x06\x43lient\x12\x13\n\x0b\x63lient_name\x18\x01 \x01(\t\"=\n\x05Reply\x12\x13\n\x0b\x63lient_name\x18\x01 \x01(\t\x12\x1f\n\x07message\x18\x02 \x01(\x0b\x32\x0e.admin.Message2\x99\x01\n\x12\x41\x64minCommunicating\x12,\n\x08Retrieve\x12\r.admin.Client\x1a\x0f.admin.Messages\"\x00\x12)\n\tSendReply\x12\x0c.admin.Reply\x1a\x0c.admin.Empty\"\x00\x12*\n\nSendResult\x12\x0c.admin.Reply\x1a\x0c.admin.Empty\"\x00\x62\x06proto3'
)




_MESSAGES = _descriptor.Descriptor(
  name='Messages',
  full_name='admin.Messages',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='admin.Messages.message', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=49,
  serialized_end=92,
)


_MESSAGE_HEADERSENTRY = _descriptor.Descriptor(
  name='HeadersEntry',
  full_name='admin.Message.HeadersEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='admin.Message.HeadersEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='admin.Message.HeadersEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=212,
  serialized_end=258,
)

_MESSAGE = _descriptor.Descriptor(
  name='Message',
  full_name='admin.Message',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='admin.Message.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='topic', full_name='admin.Message.topic', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='body_type', full_name='admin.Message.body_type', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='body', full_name='admin.Message.body', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='headers', full_name='admin.Message.headers', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_MESSAGE_HEADERSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=95,
  serialized_end=258,
)


_EMPTY = _descriptor.Descriptor(
  name='Empty',
  full_name='admin.Empty',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=260,
  serialized_end=267,
)


_CLIENT = _descriptor.Descriptor(
  name='Client',
  full_name='admin.Client',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='client_name', full_name='admin.Client.client_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=269,
  serialized_end=298,
)


_REPLY = _descriptor.Descriptor(
  name='Reply',
  full_name='admin.Reply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='client_name', full_name='admin.Reply.client_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='message', full_name='admin.Reply.message', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=300,
  serialized_end=361,
)

_MESSAGES.fields_by_name['message'].message_type = _MESSAGE
_MESSAGE_HEADERSENTRY.containing_type = _MESSAGE
_MESSAGE.fields_by_name['headers'].message_type = _MESSAGE_HEADERSENTRY
_REPLY.fields_by_name['message'].message_type = _MESSAGE
DESCRIPTOR.message_types_by_name['Messages'] = _MESSAGES
DESCRIPTOR.message_types_by_name['Message'] = _MESSAGE
DESCRIPTOR.message_types_by_name['Empty'] = _EMPTY
DESCRIPTOR.message_types_by_name['Client'] = _CLIENT
DESCRIPTOR.message_types_by_name['Reply'] = _REPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Messages = _reflection.GeneratedProtocolMessageType('Messages', (_message.Message,), {
  'DESCRIPTOR' : _MESSAGES,
  '__module__' : 'flare.private.fed.protos.admin_pb2'
  # @@protoc_insertion_point(class_scope:admin.Messages)
  })
_sym_db.RegisterMessage(Messages)

Message = _reflection.GeneratedProtocolMessageType('Message', (_message.Message,), {

  'HeadersEntry' : _reflection.GeneratedProtocolMessageType('HeadersEntry', (_message.Message,), {
    'DESCRIPTOR' : _MESSAGE_HEADERSENTRY,
    '__module__' : 'flare.private.fed.protos.admin_pb2'
    # @@protoc_insertion_point(class_scope:admin.Message.HeadersEntry)
    })
  ,
  'DESCRIPTOR' : _MESSAGE,
  '__module__' : 'flare.private.fed.protos.admin_pb2'
  # @@protoc_insertion_point(class_scope:admin.Message)
  })
_sym_db.RegisterMessage(Message)
_sym_db.RegisterMessage(Message.HeadersEntry)

Empty = _reflection.GeneratedProtocolMessageType('Empty', (_message.Message,), {
  'DESCRIPTOR' : _EMPTY,
  '__module__' : 'flare.private.fed.protos.admin_pb2'
  # @@protoc_insertion_point(class_scope:admin.Empty)
  })
_sym_db.RegisterMessage(Empty)

Client = _reflection.GeneratedProtocolMessageType('Client', (_message.Message,), {
  'DESCRIPTOR' : _CLIENT,
  '__module__' : 'flare.private.fed.protos.admin_pb2'
  # @@protoc_insertion_point(class_scope:admin.Client)
  })
_sym_db.RegisterMessage(Client)

Reply = _reflection.GeneratedProtocolMessageType('Reply', (_message.Message,), {
  'DESCRIPTOR' : _REPLY,
  '__module__' : 'flare.private.fed.protos.admin_pb2'
  # @@protoc_insertion_point(class_scope:admin.Reply)
  })
_sym_db.RegisterMessage(Reply)


_MESSAGE_HEADERSENTRY._options = None

_ADMINCOMMUNICATING = _descriptor.ServiceDescriptor(
  name='AdminCommunicating',
  full_name='admin.AdminCommunicating',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=364,
  serialized_end=517,
  methods=[
  _descriptor.MethodDescriptor(
    name='Retrieve',
    full_name='admin.AdminCommunicating.Retrieve',
    index=0,
    containing_service=None,
    input_type=_CLIENT,
    output_type=_MESSAGES,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SendReply',
    full_name='admin.AdminCommunicating.SendReply',
    index=1,
    containing_service=None,
    input_type=_REPLY,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SendResult',
    full_name='admin.AdminCommunicating.SendResult',
    index=2,
    containing_service=None,
    input_type=_REPLY,
    output_type=_EMPTY,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_ADMINCOMMUNICATING)

DESCRIPTOR.services_by_name['AdminCommunicating'] = _ADMINCOMMUNICATING

# @@protoc_insertion_point(module_scope)
