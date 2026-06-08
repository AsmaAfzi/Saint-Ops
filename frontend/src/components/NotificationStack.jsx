export default function NotificationStack({ notifications, onClose }) {
  if (!notifications.length) return null;

  return (
    <div className="notification-stack" role="status" aria-live="polite">
      {notifications.map((notification) => (
        <div
          key={notification.id}
          className={`notification notification--${notification.type}`}
        >
          <div>
            <p className="notification-title">{notification.title}</p>
            <p className="notification-message">{notification.message}</p>
          </div>
          <button
            type="button"
            className="notification-close"
            onClick={() => onClose(notification.id)}
            aria-label="Dismiss notification"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
