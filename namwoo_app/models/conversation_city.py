from sqlalchemy import Column, String
from . import Base

class ConversationCity(Base):
    __tablename__ = 'conversation_cities'
    conversation_id = Column(String(255), primary_key=True, index=True)
    city = Column(String(128), nullable=False)
