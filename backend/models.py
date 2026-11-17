import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, BigInteger, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class CompetitorProfile(Base):
    __tablename__ = "competitor_profiles"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(100), nullable=False, unique=True, index=True)
    nickname = Column(String(150), nullable=True)
    profile_url = Column(String(500), nullable=False)
    niche = Column(String(100), nullable=True, index=True)
    avatar_url = Column(String(500), nullable=True)
    notes = Column(Text, nullable=True)
    is_own_profile = Column(Boolean, default=False) # True = Meu Perfil, False = Concorrente
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    videos = relationship("VideoAsset", back_populates="competitor", cascade="all, delete-orphan")


class ProductCluster(Base):
    __tablename__ = "product_clusters"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(200), nullable=False, index=True)
    category = Column(String(100), nullable=True)
    target_audience = Column(String(250), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    videos = relationship("VideoAsset", back_populates="product_cluster")
    ideas = relationship("VideoIdea", back_populates="product_cluster", cascade="all, delete-orphan")


class VideoAsset(Base):
    __tablename__ = "video_assets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    competitor_id = Column(String(36), ForeignKey("competitor_profiles.id"), nullable=True)
    product_cluster_id = Column(String(36), ForeignKey("product_clusters.id"), nullable=True)
    tiktok_url = Column(String(500), nullable=False, index=True)
    tiktok_video_id = Column(String(50), nullable=True, index=True)
    title_caption = Column(Text, nullable=True)
    author_name = Column(String(150), nullable=True)
    author_url = Column(String(500), nullable=True)
    thumbnail_url = Column(String(1000), nullable=True)
    published_at = Column(DateTime, nullable=True)
    raw_transcript = Column(Text, nullable=True)
    status = Column(String(30), default="pending_analysis") # pending_analysis, analyzed, error
    created_at = Column(DateTime, default=datetime.utcnow)

    competitor = relationship("CompetitorProfile", back_populates="videos")
    product_cluster = relationship("ProductCluster", back_populates="videos")
    snapshots = relationship("MetricSnapshot", back_populates="video", cascade="all, delete-orphan", order_by="desc(MetricSnapshot.recorded_at)")
    analysis = relationship("VideoAnalysis", back_populates="video", uselist=False, cascade="all, delete-orphan")


class MetricSnapshot(Base):
    __tablename__ = "metric_snapshots"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("video_assets.id"), nullable=False, index=True)
    views = Column(BigInteger, nullable=True)
    likes = Column(BigInteger, nullable=True)
    comments = Column(BigInteger, nullable=True)
    shares = Column(BigInteger, nullable=True)
    extraction_source = Column(String(30), default="manual") # manual, oembed, yt_dlp
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)

    video = relationship("VideoAsset", back_populates="snapshots")


class VideoAnalysis(Base):
    __tablename__ = "video_analyses"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    video_id = Column(String(36), ForeignKey("video_assets.id"), nullable=False, unique=True, index=True)
    hook_initial = Column(Text, nullable=True)
    headline = Column(Text, nullable=True)
    pain_or_desire = Column(Text, nullable=True)
    sales_argument = Column(Text, nullable=True)
    video_structure = Column(String(100), nullable=True)
    product_demo_style = Column(Text, nullable=True)
    cta_type = Column(Text, nullable=True)
    hook_strength_score = Column(Float, nullable=True)
    key_takeaways = Column(Text, nullable=True)
    raw_analysis_json = Column(Text, nullable=True)
    analyzed_at = Column(DateTime, default=datetime.utcnow)

    video = relationship("VideoAsset", back_populates="analysis")


class VideoIdea(Base):
    __tablename__ = "video_ideas"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    product_cluster_id = Column(String(36), ForeignKey("product_clusters.id"), nullable=False, index=True)
    based_on_video_ids = Column(Text, nullable=True) # JSON list of video IDs
    angle_title = Column(String(250), nullable=False)
    hook_suggestion = Column(Text, nullable=False)
    proposed_structure = Column(Text, nullable=True)
    differentiation_angle = Column(Text, nullable=False)
    script_outline = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    product_cluster = relationship("ProductCluster", back_populates="ideas")


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(150), nullable=False)
    template_type = Column(String(50), nullable=False, index=True) # video_analysis, pattern_matching, idea_generation
    system_prompt = Column(Text, nullable=True)
    template_content = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
